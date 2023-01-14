#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	string modelpath;
	string datatype;
};

class FreeYOLO
{
public:
	FreeYOLO(Net_config config);
	void detect(Mat& frame);
private:
	int inpWidth;
	int inpHeight;
	int nout;
	int num_proposal;
	vector<string> class_names;
	int num_class;
	const int num_stride = 3;
	int strides[3] = { 8,16,32 };

	float confThreshold;
	float nmsThreshold;
	Net net;
};

FreeYOLO::FreeYOLO(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;

	this->net = readNet(config.modelpath);

	size_t pos = config.modelpath.rfind("_");
	size_t pos_ = config.modelpath.rfind(".");
	int len = pos_ - pos - 1;
	string hxw = config.modelpath.substr(pos + 1, len);
	pos = hxw.rfind("x");
	string h = hxw.substr(0, pos);
	len = hxw.length() - pos;
	string w = hxw.substr(pos + 1, len);
	this->inpHeight = stoi(h);
	this->inpWidth = stoi(w);

	if (config.datatype == "coco")
	{
		string classesFile = "coco.names";
		ifstream ifs(classesFile.c_str());
		string line;
		while (getline(ifs, line)) this->class_names.push_back(line);
	}
	else if (config.datatype == "face")
	{
		this->class_names.push_back("face");
	}
	else
	{
		this->class_names.push_back("person");

	}
	this->num_class = class_names.size();
}

void FreeYOLO::detect(Mat& frame)
{
	const float ratio = std::min(float(this->inpHeight) / float(frame.rows), float(this->inpWidth) / float(frame.cols));
	const int neww = int(frame.cols * ratio);
	const int newh = int(frame.rows * ratio);

	Mat dstimg;
	resize(frame, dstimg, Size(neww, newh));
	copyMakeBorder(dstimg, dstimg, 0, this->inpHeight - newh, 0, this->inpWidth - neww, BORDER_CONSTANT, 114);

	Mat blob = blobFromImage(dstimg);
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());   // 开始推理

	num_proposal = outs[0].size[1];
	nout = outs[0].size[2];
	const float* pdata = (float*)outs[0].data;
	int n = 0, i = 0, j = 0, k = 0; ///cx, cy, w, h, box_score, class_score
	vector<float> confidences;
	vector<Rect> boxes;
	vector<int> classIds;
	for (n = 0; n < this->num_stride; n++)   ///特征图尺度
	{
		int num_grid_x = (int)ceil((this->inpWidth / strides[n]));
		int num_grid_y = (int)ceil((this->inpHeight / strides[n]));
		for (i = 0; i < num_grid_y; i++)
		{
			for (j = 0; j < num_grid_x; j++)
			{
				const float box_score = pdata[4];
				int max_ind = 0;
				float max_class_socre = 0;
				for (k = 0; k < num_class; k++)
				{
					if (pdata[k + 5] > max_class_socre)
					{
						max_class_socre = pdata[k + 5];
						max_ind = k;
					}
				}
				max_class_socre *= box_score;
				max_class_socre = sqrt(max_class_socre);

				if (max_class_socre > this->confThreshold)
				{
					float cx = (0.5f + j + pdata[0]) * strides[n];  ///cx
					float cy = (0.5f + i + pdata[1]) * strides[n];   ///cy
					float w = expf(pdata[2]) * strides[n];   ///w
					float h = expf(pdata[3]) * strides[n];  ///h

					float xmin = (cx - 0.5 * w) / ratio;
					float ymin = (cy - 0.5 * h) / ratio;
					float xmax = (cx + 0.5 * w) / ratio;
					float ymax = (cy + 0.5 * h) / ratio;

					int left = int((cx - 0.5 * w) / ratio);
					int top = int((cy - 0.5 * h) / ratio);
					int width = int(w / ratio);
					int height = int(h / ratio);

					confidences.push_back(max_class_socre);
					boxes.push_back(Rect(left, top, width, height));
					classIds.push_back(max_ind);
				}
				pdata += nout;
			}
		}		
	}

	vector<int> indices;
	dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);

	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(0, 0, 255), 3);
		string label = format("%.2f", confidences[idx]);
		label = this->class_names[classIds[idx]] + ":" + label;
		putText(frame, label, Point(box.x, box.y - 10), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
	}
}

int main()
{
	Net_config cfg = { 0.8, 0.5, "weights/face/yolo_free_huge_widerface_192x320.onnx", "face" };
	FreeYOLO net(cfg);
	string imgpath = "images/face/1.jpg";
	Mat srcimg = imread(imgpath);
	net.detect(srcimg);

	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}