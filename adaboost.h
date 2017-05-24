#ifndef _ADABOOST_H_
#define _ADABOOST_H_ 
#define MAX_FEATURE 100
#define MAX_SAMPLES 500
//#define DEBUG  
struct Sample
{
	double weight;
	double feature[MAX_FEATURE];
	int    indicate;
};
struct SampleHeader
{
	int samplesNum;
	int featureNum;

	//double feature[MAX_SAMPLES][MAX_FEATURE];
	struct Sample samples[MAX_SAMPLES];

};

struct Stump
{
	int left;
	int right;
	double alpha;
	int fIdx;
	double ft;
};

struct Classifier
{
	struct Stump stump;
	struct Classifier* next;
};


struct ClassifierHeader
{
	int classifierNum;
	struct Classifier* classifier;
};
struct IdxHeader
{
	int samplesNum;
	int featureNum;

	double feature[MAX_FEATURE][MAX_SAMPLES];

};

#endif