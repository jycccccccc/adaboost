#include "stdio.h"
#include "assert.h"
#include "string.h"
#include "stdlib.h"
#include "math.h"
#include "adaboost.h"

#define DATA_NAME "..//dataset//ionosphere.data"

SampleHeader sampleHeader;
IdxHeader idx;


//==================================================================
//sort
//==================================================================
void sort(double a[], int n)
{
	double tmp;
	for (int i = 0; i < n - 1; i++)
	{
		for (int j = 0; j < n - i - 1; j++)
		{
			if (a[j] > a[j + 1])
			{
				tmp = a[j];
				a[j] = a[j + 1];
				a[j + 1] = tmp;
			}
		}
	}
}

//==================================================================
//countFeature
//==================================================================
int countFeature(const char* buf)
{
	const char* p = buf;
	int	  cnt = 0;
	while (*p != NULL)
	{
		if (*p == ',')
			cnt++;
		p++;
	}

	return cnt;
}

//==================================================================
//setFeature
//==================================================================
void setFeature(char* buf)
{
	int i = 0;
	struct Sample sample;

	//char*p = strtok(buf, ",");
	char *ptr;
	char *p;
	ptr = strtok_s(buf,",",&p);

	sample.feature[i++] = atof(p);

	while (1)
	{
		if (*p != 'g' && *p != 'b')
			sample.feature[i++] = atof(p);

		else
			break;

		//p = strtok(NULL, ",");
		ptr = strtok_s(NULL, ",", &p);
	}

	if (*p == 'g')
		sample.indicate = 1;
	else if (*p == 'b')
		sample.indicate = -1;
	else
		assert(0);

	for (i = 0; i < sampleHeader.featureNum; i++)
		idx.feature[i][idx.samplesNum] = sample.feature[i];

	sampleHeader.samples[sampleHeader.samplesNum] = sample;
	sampleHeader.samplesNum++;
	idx.samplesNum++;

};


//==================================================================
//loadData
//==================================================================
void loadData()
{
	FILE *fp = NULL;
	char buf[1000];
	int featureCnt = 0;
	double* featrue = NULL;
	double* featruePtr = NULL;
	int i = 0;
	int n = 0;

	fopen_s(&fp, DATA_NAME, "r");


	fgets(buf, 1000, fp);
	idx.featureNum = sampleHeader.featureNum = countFeature(buf);

	setFeature(buf);

	while (!feof(fp))
	{
	
		fgets(buf, 1000, fp);
		setFeature(buf);
	
	}

	fclose(fp);

	for (i = 0; i < idx.featureNum; i++)
		sort(idx.feature[i], idx.samplesNum);





}


//==================================================================
//CreateStump
//==================================================================
Stump CreateStump()
{
	int i, j, k;
	Stump stump;
	double min = 0xffffffff;
	double err = 0;
	double flipErr = 0;

	double feature;
	int indicate;
	double weight;
	double pre;
	for (i = 0; i < idx.featureNum; i++)
	{
		pre = 0xffffffff;
		for (j = 0; j < idx.samplesNum; j++)
		{

			err = 0;
			double rootFeature = idx.feature[i][j];

			if (pre == rootFeature)
				continue;


			for (k = 0; k < sampleHeader.samplesNum; k++)
			{
				feature = sampleHeader.samples[k].feature[i];
				indicate = sampleHeader.samples[k].indicate;
				weight = sampleHeader.samples[k].weight;
				if ((feature <  rootFeature  && indicate != 1) || \
					(feature >= rootFeature && indicate != -1)
					)
					err += weight;

			}

			flipErr = 1 - err;
			err = err < flipErr ? err : flipErr;

			if (err < min)
			{
				min = err;
				stump.fIdx = i;
				stump.ft = rootFeature;
				if (err < flipErr)
				{
					stump.left = 1;
					stump.right = -1;
				}
				else
				{

					stump.left = -1;
					stump.right = 1;
				}
			}

			pre = rootFeature;
		}
	}


	stump.alpha = 0.5*log(1.0 / min - 1);
	return stump;
}


//==================================================================
//reSetWeight
//==================================================================
void reSetWeight(struct Stump stump)
{
	int i;
	double z = 0;

	for (i = 0; i < sampleHeader.samplesNum; i++)
	{
		double feature = (sampleHeader.samples[i]).feature[stump.fIdx];
		double rs = feature < stump.ft ? stump.left : stump.right;
		rs = stump.alpha * rs * sampleHeader.samples[i].indicate;

		z += sampleHeader.samples[i].weight * exp(-1.0 * rs);
	}



	for (i = 0; i < sampleHeader.samplesNum; i++)
	{
		double feature = (sampleHeader.samples[i]).feature[stump.fIdx];
		double rs = feature < stump.ft ? stump.left : stump.right;
		rs = stump.alpha * rs * sampleHeader.samples[i].indicate;

		sampleHeader.samples[i].weight = sampleHeader.samples[i].weight * exp(-1.0 * rs) / z;


	}
}

//==================================================================
//AdaBoost
//==================================================================
void AdaBoost(int interation)
{
	int i;
	struct ClassifierHeader head;
	struct Classifier* pCls = NULL;
	struct Classifier* tmp = NULL;
	head.classifierNum = interation;

	loadData();

	for (i = 0; i < sampleHeader.samplesNum; i++)
		sampleHeader.samples[i].weight = 1.0 / sampleHeader.samplesNum;


	head.classifier = (struct Classifier*)malloc(sizeof(struct Classifier));
	pCls = head.classifier;
	pCls->stump = CreateStump();
	reSetWeight(pCls->stump);
	printf("+-----------+--+-------+\n");
	printf("|   alpha   |id|  ft   |\n");
	printf("+-----------+--+-------+\n");
	printf("|%.9lf|%2d|%+.4lf|\n", pCls->stump.alpha, pCls->stump.fIdx, pCls->stump.ft);
	printf("+-----------+--+-------+\n");
	for (i = 1; i < head.classifierNum; i++)
	{

		pCls = pCls->next = (struct Classifier*)malloc(sizeof(struct Classifier));

		pCls->stump = CreateStump();
		reSetWeight(pCls->stump);
		printf("|%.9lf|%2d|%+.4lf|\n", pCls->stump.alpha, pCls->stump.fIdx, pCls->stump.ft);
		printf("+-----------+--+-------+\n");

	}


	printf("\n");

	for (i = 0, pCls = head.classifier; i < head.classifierNum; i++)
	{
		tmp = pCls;
		pCls = tmp->next;
		free(tmp);
	}

}


void main()
{
	printf("begin\n");
	AdaBoost(100);
	printf("end\n");
}