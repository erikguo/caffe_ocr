
#pragma once


#ifndef interface
#define CINTERFACE
#define interface struct
#endif

//-------------------------------------------------------------------------------------------------
#ifndef IMPORT
#define IMPORT __declspec(dllimport)
#endif


//-------------------------------------------------------------------------------------------------
#ifndef EXPORT
#define EXPORT __declspec(dllexport)
#endif

#include <vector>
#include <string>

typedef std::pair<std::string, float> Prediction;
typedef std::pair<int, float> PredictionIdx;

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>




 interface ICNNPredict
{
	 virtual bool IsCPUMode() = 0;
	virtual std::vector<Prediction> Classify(const std::string& file, int N = 5) = 0;
	virtual std::vector<Prediction> Classify(const unsigned char* pJPGBuffer, int len, int N = 5) = 0;
	virtual std::vector<Prediction> Classify(const cv::Mat& img, int N = 5) = 0;
	virtual std::vector<std::vector<PredictionIdx> > BatchClassify(const std::vector<cv::Mat>& imgs, int N = 5) = 0;
	virtual std::vector<std::vector<float> > BatchPredict(const std::vector<cv::Mat>& img)=0;
	virtual std::vector<Prediction> CropClassify(const cv::Mat& img, int std_size, int crop_num = 1, int N = 5) = 0;
	virtual std::vector<PredictionIdx> ClassifyRtnIdx(const cv::Mat& img, int N = 5) = 0;

	//virtual std::vector<float> ExtractFeature(const cv::Mat& img, const std::string& strLayerName = "") = 0;
	virtual std::vector<float> GetLayerFeatureMaps(const std::string& strLayerName, std::vector<int>& outshape) = 0;
	virtual int GetFeatureDim() = 0;
	virtual std::vector< std::vector<float> > GetLastBlockFeature(const cv::Mat& img) = 0;
	virtual std::vector<float> GetOutputFeatureMap(const cv::Mat& img, std::vector<int>& outshape) = 0;

	virtual std::vector<std::string> GetLabels() = 0;

	virtual int GetLabelSize() = 0;
	virtual void SetMean(const std::string& mean_file) = 0;

	virtual std::vector<float> Predict(const cv::Mat& img) = 0;
	virtual void GetInputImageSize(int &w, int &h) = 0;

	//advanced operations
	virtual float Pruning(float weight_t, const char* saveas_name=0)=0;
	virtual cv::Mat EstimateReceptiveField(const cv::Mat& img, const std::string& layerName, int x, int y, int idxNeuron = -1,bool islstm=false,int* width_parts=0) = 0;
	virtual void GetLayerFeatureMapSize(int w, int h, const std::string& layerName,int& w1, int& h1)=0;
	virtual void Release()=0;

	virtual void InitLexicon(const char* lexicon_file = 0, bool is_wcs = false) = 0;
	virtual const char* GetOutputFeatureMapByLexicon(const cv::Mat& img, bool is_wcs = false) = 0;
};

 typedef unsigned char byte;

 extern "C" 
 {
	 EXPORT ICNNPredict* CreatePredictInstance(const char* model_folder, bool use_gpu, int gpu_no);
	 EXPORT void ICNNPredict_InitLexicon(ICNNPredict* pcnn, const char* lexicon_file) {
		 pcnn->InitLexicon(lexicon_file);
	 }
	 EXPORT void ICNNPredict_GetLabels(ICNNPredict* pcnn, char** labelsbuf) {
		 std::vector<std::string> labels = pcnn->GetLabels();
		 for (int n = 0; n < labels.size(); n++) {
			 labelsbuf[n] = new char[3];
			 labelsbuf[n][2] = '\0';
			 strncpy(labelsbuf[n], labels[n].c_str(), 2);
		 } 
	 }
	 EXPORT int ICNNPredict_GetLabelSize(ICNNPredict* pcnn) { return pcnn->GetLabelSize(); }
	 EXPORT void ICNNPredict_GetInputImageSize(ICNNPredict* pcnn, int &w, int &h) { pcnn->GetInputImageSize(w, h); }
	 EXPORT int ICNNPredict_GetOutputFeatureMap(ICNNPredict* pcnn, int rows, int cols, int channels, byte* data, float *pred) {
		 std::vector<int> outshape(4);
		 int size[3] = {rows, cols, channels};
		 const cv::Mat img = cv::Mat(rows, cols, CV_8UC3, data);
		 std::vector<float> v_pred = pcnn->GetOutputFeatureMap(img, outshape);

		 int num = v_pred.size();
		 if (num > 0) {
			 memcpy(pred, &v_pred[0], num * sizeof(float));
		 }
		 return num;
	 }
	 EXPORT const char* ICNNPredict_GetOutputFeatureMapByLexicon(ICNNPredict* pcnn, int rows, int cols, int channels, byte* data) {
		 int size[3] = { rows, cols, channels };
		 const cv::Mat img = cv::Mat(rows, cols, CV_8UC3, data);
		 const char* v_pred = pcnn->GetOutputFeatureMapByLexicon(img);

		 return v_pred;
	 }
 }