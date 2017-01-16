//#include "SlicCudaHost.h"
//#include "SlicCudaDevice.h"

using namespace std;
using namespace cv;


SlicCuda::SlicCuda(){
	int nbGpu = 0;
	gpuErrchk(cudaGetDeviceCount(&nbGpu));
	cout << "Detected " << nbGpu << " cuda capable gpu" << endl;
	m_deviceId = 0;
	gpuErrchk(cudaSetDevice(m_deviceId));
	gpuErrchk(cudaGetDeviceProperties(&m_deviceProp, m_deviceId));

}

SlicCuda::~SlicCuda(){
	delete[] h_fClusters;
	delete[] h_fLabels;
	gpuErrchk(cudaFree(d_fClusters));
	gpuErrchk(cudaFree(d_fAccAtt));
	gpuErrchk(cudaFreeArray(cuArrayFrameBGRA));
	gpuErrchk(cudaFreeArray(cuArrayFrameLab));
	gpuErrchk(cudaFreeArray(cuArrayLabels));
}

void SlicCuda::initialize(const cv::Mat& frame0, const int diamSpxOrNbSpx , const InitType initType, const float wc , const int nbIteration ) {
	m_nbIteration = nbIteration;
	m_FrameWidth = frame0.cols;
	m_FrameHeight = frame0.rows;
	m_nbPx = m_FrameWidth*m_FrameHeight;
	m_InitType = initType;
	m_wc = wc;
	if (m_InitType == SLIC_NSPX){
		m_SpxDiam = diamSpxOrNbSpx; 
		m_SpxDiam = (int)sqrt(m_nbPx / (float)diamSpxOrNbSpx);
	}
	else m_SpxDiam = diamSpxOrNbSpx;
	
	getSpxSizeFromDiam(m_FrameWidth, m_FrameHeight, m_SpxDiam, &m_SpxWidth, &m_SpxHeight); // determine w and h of Spx based on diamSpx
	m_SpxArea = m_SpxWidth*m_SpxHeight;
	CV_Assert(m_nbPx%m_SpxArea == 0); 
	m_nbSpx = m_nbPx / m_SpxArea; 

	h_fClusters = new float[m_nbSpx * 5]; // m_nbSpx * [L,a,b,x,y]
	h_fLabels = new float[m_nbPx];

	initGpuBuffers();
}

void SlicCuda::segment(const Mat& frameBGR) {
	uploadFrame(frameBGR);
	

	
	gpuRGBA2Lab();



	gpuInitClusters();
	cudaMemcpy(h_fClusters, d_fClusters,m_nbSpx*5*sizeof(float), cudaMemcpyDeviceToHost);
	Mat tmpCluster(1, m_nbSpx * 5, CV_32F, h_fClusters);
	cout << tmpCluster << endl;


	for (int i = 0; i<m_nbIteration; i++) {
		assignment();
		cudaDeviceSynchronize();
		update();
		cudaDeviceSynchronize();
	}
	downloadLabels();
}

void SlicCuda::initGpuBuffers() {
	//allocate buffers on gpu

	cudaChannelFormatDesc channelDescrBGRA = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	gpuErrchk(cudaMallocArray(&cuArrayFrameBGRA, &channelDescrBGRA, m_FrameWidth, m_FrameHeight));

	cudaChannelFormatDesc channelDescrLab = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	gpuErrchk(cudaMallocArray(&cuArrayFrameLab, &channelDescrLab, m_FrameWidth, m_FrameHeight, cudaArraySurfaceLoadStore));

	cudaChannelFormatDesc channelDescrLabels = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	gpuErrchk(cudaMallocArray(&cuArrayLabels, &channelDescrLabels, m_FrameWidth, m_FrameHeight, cudaArraySurfaceLoadStore));



	/*
#if __CUDA_ARCH__>=300
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = frameBGRA_array;

	// Specify texture object parameters
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = false;
	gpuErrchk(cudaCreateTextureObject(&frameBGRA_tex, &resDesc, &texDesc, NULL));
#else*/
	texFrameBGRA.addressMode[0] = cudaAddressModeClamp;
	texFrameBGRA.addressMode[1] = cudaAddressModeClamp;
	texFrameBGRA.filterMode = cudaFilterModePoint;
	texFrameBGRA.normalized = false;
	cudaBindTextureToArray(&texFrameBGRA, cuArrayFrameBGRA, &channelDescrBGRA);
//#endif

	// surface frameLab
	
/*#if __CUDA_ARCH__>=300
	cudaResourceDesc resDescLab;
	memset(&resDescLab, 0, sizeof(resDescLab));
	resDescLab.resType = cudaResourceTypeArray;

	resDescLab.res.array.array = frameLab_array;
	gpuErrchk(cudaCreateSurfaceObject(&frameLab_surf, &resDescLab));
#else*/
	cudaBindSurfaceToArray(&surfFrameLab, cuArrayFrameLab, &channelDescrLab);
	
//#endif

	// surface labels
	/*
#if __CUDA_ARCH__>=300

	cudaResourceDesc resDescLabels;
	memset(&resDescLabels, 0, sizeof(resDescLabels));
	resDescLabels.resType = cudaResourceTypeArray;

	resDescLabels.res.array.array = labels_array;
	gpuErrchk(cudaCreateSurfaceObject(&labels_surf, &resDescLabels));

#else*/
	gpuErrchk(cudaBindSurfaceToArray(&surfLabels, cuArrayLabels, &channelDescrLabels));
//#endif
	
	// buffers clusters , accAtt
	gpuErrchk(cudaMalloc((void**)&d_fClusters, m_nbSpx*sizeof(float) * 5)); // 5-D centroid
	gpuErrchk(cudaMalloc((void**)&d_fAccAtt, m_nbSpx*sizeof(float) * 6)); // 5-D centroid acc + 1 counter
	cudaMemset(d_fAccAtt, 0, m_nbSpx*sizeof(float) * 6);//initialize accAtt to 0
	
}


void SlicCuda::uploadFrame(const Mat& frameBGR) { 
	cv::Mat frameBGRA;
	cv::cvtColor(frameBGR, frameBGRA, CV_BGR2BGRA);
	CV_Assert(frameBGRA.type() == CV_8UC4);
	CV_Assert(frameBGRA.isContinuous());
	gpuErrchk(cudaMemcpyToArray(cuArrayFrameBGRA, 0, 0, (uchar*)frameBGRA.data, m_nbPx* sizeof(uchar4), cudaMemcpyHostToDevice));
	
	/*uchar* pfTmpFrameBGRA = new uchar[4 * m_nbPx];
	cudaMemcpyFromArray(pfTmpFrameBGRA, cuArrayFrameBGRA, 0, 0, sizeof(uchar) * 4 * m_nbPx, cudaMemcpyDeviceToHost);
	Mat tmpFrameBGRA(frameBGR.size(), CV_8UC4, pfTmpFrameBGRA);
	cout << tmpFrameBGRA << endl;*/
}

void SlicCuda::gpuRGBA2Lab() {
	const int blockW = 8; 
	const int blockH = blockW;
	CV_Assert(blockW*blockH <= m_deviceProp.maxThreadsPerBlock);
	dim3 threadsPerBlock(blockW, blockH);
	dim3 numBlocks(iDivUp(m_FrameWidth, blockW), iDivUp(m_FrameHeight, blockH));


	Mat somV(m_FrameHeight, m_FrameWidth, CV_32FC4, Scalar(2, 3, 4, 5));
	gpuErrchk(cudaMemcpyToArray(cuArrayFrameLab, 0, 0, (float*)somV.data, m_nbPx* sizeof(float4), cudaMemcpyHostToDevice));


	kRgb2CIELab << <numBlocks, threadsPerBlock >> >(m_FrameWidth, m_FrameHeight);

	cudaDeviceSynchronize();

	float* pfTmpLab = new float[4 * m_nbPx];
	gpuErrchk(cudaMemcpyFromArray(pfTmpLab, cuArrayFrameLab, 0, 0, sizeof(float) * 4 * m_nbPx, cudaMemcpyDeviceToHost));
	Mat tmpLab(m_FrameHeight, m_FrameWidth, CV_32FC4, pfTmpLab);
	//cout <<"LabFrame"<< tmpLab << endl;
	
}



void SlicCuda::gpuInitClusters() {
	int blockW = 16;
	dim3 threadsPerBlock(blockW);
	dim3 numBlocks(iDivUp(m_nbSpx, blockW));

	kInitClusters << <numBlocks, threadsPerBlock >> >(m_FrameWidth, m_FrameHeight, m_FrameWidth / m_SpxWidth, m_FrameHeight / m_SpxHeight, d_fClusters);


}

void SlicCuda::assignment(){
	int hMax = m_deviceProp.maxThreadsPerBlock / m_SpxHeight;
	int nBlockPerClust = iDivUp(m_SpxHeight, hMax);

	dim3 blockPerGrid(m_nbSpx, nBlockPerClust);
	dim3 threadPerBlock(m_SpxWidth, std::min(m_SpxHeight, hMax));

	CV_Assert(threadPerBlock.x >= 3 && threadPerBlock.y >= 3);

	float wc2 = m_wc * m_wc;

	kAssignment << < blockPerGrid, threadPerBlock >> >(m_FrameWidth, m_FrameHeight, m_SpxWidth, m_SpxHeight, wc2, d_fClusters, d_fAccAtt);

}

void SlicCuda::update(){
	dim3 threadsPerBlock(m_deviceProp.maxThreadsPerBlock);
	dim3 numBlocks(iDivUp(m_nbSpx, m_deviceProp.maxThreadsPerBlock));
	kUpdate << <numBlocks, threadsPerBlock >> >(m_nbSpx, d_fClusters, d_fAccAtt);
}

void SlicCuda::downloadLabels()
{
	cudaMemcpyFromArray(h_fLabels, cuArrayLabels, 0, 0, m_nbPx* sizeof(float), cudaMemcpyDeviceToHost);
}

void SlicCuda::displayBound(cv::Mat& image, const float* labels, const cv::Scalar colour)
{
	//load label from gpu

	const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	/* Initialize the contour vector and the matrix detailing whether a pixel
	* is already taken to be a contour. */
	vector<cv::Point> contours;
	vector<vector<bool> > istaken;
	for (int i = 0; i < image.rows; i++) {
		vector<bool> nb;
		for (int j = 0; j < image.cols; j++) {
			nb.push_back(false);
		}
		istaken.push_back(nb);
	}

	/* Go through all the pixels. */

	for (int i = 0; i<image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {

			int nr_p = 0;

			/* Compare the pixel to its 8 neighbours. */
			for (int k = 0; k < 8; k++) {
				int x = j + dx8[k], y = i + dy8[k];

				if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
					if (istaken[y][x] == false && labels[i*image.cols + j] != labels[y*image.cols + x]) {
						nr_p += 1;
					}
				}
			}
			/* Add the pixel to the contour list if desired. */
			if (nr_p >= 2) {
				contours.push_back(cv::Point(j, i));
				istaken[i][j] = true;
			}

		}
	}

	/* Draw the contour pixels. */
	for (int i = 0; i < (int)contours.size(); i++) {
		image.at<cv::Vec3b>(contours[i].y, contours[i].x) = cv::Vec3b(colour[0], colour[1], colour[2]);
	}
}
