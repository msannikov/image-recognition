#include "util.h"

const char *dataFileName[10] = {"data/data0", "data/data1", "data/data2", "data/data3", "data/data4", "data/data5", "data/data6", "data/data7", "data/data8", "data/data9"};

const float EPS = 1e-9;
const int STUDYN = 200;
float ETA = 0.005;
//---------------------------------------------------------------------------

int k1[25] = {0, 1, 2, 3, 4,
    29, 30, 31, 32, 33,
    58, 59, 60, 61, 62,
    87, 88, 89, 90, 91,
    116, 117, 118, 119, 120};

int k2[25] = {0, 1, 2, 3, 4,
    13, 14, 15, 16, 17,
    26, 27, 28, 29, 30,
    39, 40, 41, 42, 43,
    52, 53, 54, 55, 56};

float f(float s)
{
    return 1.7159 * tanhf(0.66666667 * s);
}

float df(float s)
{
    return 1.7159 * 0.66666667 * (1. - sqr(0.66666667 * s));
}

void exec1(float *neur1, float *w1, float *neur2)
{
	for(int tmap = 0; tmap < 6; ++tmap)
	{
		for(int x = 0; x < 13; ++x)
		{
			for(int y = 0; y < 13; ++y)
			{
				int ind = tmap * 26;
				float res = w1[ind++];
				for(int i = 0; i < 25; ++i, ++ind)
					res += neur1[2 * 29 * y + 2 * x + k1[i]] * w1[ind];
				neur2[169 * tmap + y * 13 + x] = f(res);
			}
		}
	}
}

void exec2(float *neur2, float *w2, float *neur3)
{
	for(int tmap = 0; tmap < 50; ++tmap)
	{
		for(int x = 0; x < 5; ++x)
		{
			for(int y = 0; y < 5; ++y)
			{
				int ind = tmap * 156;
				float res = w2[ind++];
				for(int i = 0; i < 25; ++i)
				{
                    for(int j = 0; j < 6; ++j)
                        res += neur2[13 * 13 * j + 2 * x + 13 * 2 * y + k2[i]] *
                        w2[ind + i * 6 + j];
                }
				neur3[25 * tmap + y * 5 + x] = f(res);
			}
		}
	}
}

void exec3(float *neur3, float *w3, float *neur4)
{
	for(int tmap = 0; tmap < 100; ++tmap)
	{
		int ind = tmap * 1251;
		float res = w3[ind++];
		for(int i = 0; i < 1250; ++i, ++ind)
			res += neur3[i] * w3[ind];
		neur4[tmap] = f(res);
	}
}

void exec4(float *neur4, float *w4, float *neur5)
{
	for(int tmap = 0; tmap < 10; ++tmap)
	{
		int ind = tmap * 101;
		float res = w4[ind++];
		for(int i = 0; i < 100; ++i, ++ind)
			res += neur4[i] * w4[ind];
		neur5[tmap] = f(res);
	}
}

void calculate(float *w[4], float *x[5])
{
	exec1(x[0], w[0], x[1]);
	exec2(x[1], w[1], x[2]);
	exec3(x[2], w[2], x[3]);
	exec4(x[3], w[3], x[4]);
}
//------------------------------------------------------------

void calcNeuronErrors(float *x[5], float *w[4], float *y, float *neuronError[5])
{
    for(int i = 0; i < 10; ++i)
        neuronError[4][i] = df(x[4][i]) * (x[4][i] - y[i]);
    
    for(int tmap = 0; tmap < 10; ++tmap)
	{
		int ind = tmap * 101;
		++ind;
		for(int i = 0; i < 100; ++i, ++ind)
            neuronError[3][i] += neuronError[4][tmap] * w[3][ind];
    }
    for(int i = 0; i < 100; ++i)
        neuronError[3][i] *= df(x[3][i]);
    
    for(int tmap = 0; tmap < 100; ++tmap)
	{
		int ind = tmap * 1251;
		++ind;
		for(int i = 0; i < 1250; ++i)
			neuronError[2][i] += neuronError[3][tmap] * w[2][ind++];
	}
    for(int i = 0; i < 1250; ++i)
        neuronError[2][i] *= df(x[2][i]);
    
    for(int tmap = 0; tmap < 50; ++tmap)
	{
		for(int x = 0; x < 5; ++x)
		{
			for(int y = 0; y < 5; ++y)
			{
				int ind = tmap * 156;
				++ind;
				for(int i = 0; i < 25; ++i)
				{
                    for(int j = 0; j < 6; ++j)
                        neuronError[1][13 * 13 * j + 2 * x + 13 * 2 * y + k2[i]] +=
                        neuronError[2][25 * tmap + y * 5 + x] * w[1][ind + i * 6 + j];
                }
			}
		}
	}
    for(int i = 0; i < 1014; ++i)
        neuronError[1][i] *= df(x[1][i]);
    
    for(int tmap = 0; tmap < 6; ++tmap)
	{
		for(int x = 0; x < 13; ++x)
		{
			for(int y = 0; y < 13; ++y)
			{
				int ind = tmap * 26;
				++ind;
				for(int i = 0; i < 25; ++i, ++ind)
                    neuronError[0][2 * 29 * y + 2 * x + k1[i]] +=
                    neuronError[1][169 * tmap + y * 13 + x] * w[0][ind];
			}
		}
	}
    for(int i = 0; i < 841; ++i)
        neuronError[0][i] *= df(x[0][i]);
}

void calcNewWeights(float *w[4], float *neur[5], float *neuronError[5])
{
    for(int tmap = 0; tmap < 6; ++tmap)
	{
		for(int x = 0; x < 13; ++x)
		{
			for(int y = 0; y < 13; ++y)
			{
                float curError = neuronError[1][169 * tmap + y * 13 + x];
                
				int ind = tmap * 26;
                w[0][ind] += -ETA * curError; // bias
                ++ind;
                
				for(int i = 0; i < 25; ++i, ++ind)
                    w[0][ind] += -ETA * curError * neur[0][2 * 29 * y + 2 * x + k1[i]];
			}
		}
	}
    
    for(int tmap = 0; tmap < 50; ++tmap)
	{
		for(int x = 0; x < 5; ++x)
		{
			for(int y = 0; y < 5; ++y)
			{
                float curError = neuronError[2][25 * tmap + y * 5 + x];
                
				int ind = tmap * 156;
				w[1][ind] += -ETA * curError; // bias
                ++ind;
                
				for(int i = 0; i < 25; ++i)
				{
                    for(int j = 0; j < 6; ++j)
                    {
                        w[1][ind + i * 6 + j] += -ETA * curError *
                        neur[1][13 * 13 * j + 2 * x + 13 * 2 * y + k2[i]];
                    }
                }
			}
		}
	}
    
    for(int tmap = 0; tmap < 100; ++tmap)
	{
        float curError = neuronError[3][tmap];
        
		int ind = tmap * 1251;
		w[2][ind] += -ETA * curError; // bias
        ++ind;
        
		for(int i = 0; i < 1250; ++i, ++ind)
            w[2][ind] += -ETA * curError * neur[2][i];
	}
    
    for(int tmap = 0; tmap < 10; ++tmap)
	{
		int ind = tmap * 101;
        float curError = neuronError[4][tmap];
        
        w[3][ind] += -ETA * curError; // bias
        ++ind;
        
		for(int i = 0; i < 100; ++i, ++ind)
            w[3][ind] += -ETA * curError * neur[3][i];
	}
}

void backpropagate(float *w[4], float *x[5], float *y)
{
    float *neuronError[5];
    
    neuronError[0] = new float[841];
    neuronError[1] = new float[1014];
    neuronError[2] = new float[1250];
    neuronError[3] = new float[100];
    neuronError[4] = new float[10];
    
    memset(neuronError[0], 0, sizeof(float) * 841);
    memset(neuronError[1], 0, sizeof(float) * 1014);
    memset(neuronError[2], 0, sizeof(float) * 1250);
    memset(neuronError[3], 0, sizeof(float) * 100);
    memset(neuronError[4], 0, sizeof(float) * 10);
    
    calcNeuronErrors(x, w, y, neuronError);
    calcNewWeights(w, x, neuronError);
}

float getNetError(float *output, int size, float *y)
{
    float res = 0;
    for(int i = 0; i < size; ++i)
        res += sqr(output[i] - y[i]);
    return res / size;
}

void getInputLayer(float *inputLayer, FILE *f)
{
    uchar a[29][29];
    memset(a, 0, sizeof a);
    for(int i = 0; i < 28; ++i)
        fread(&a[i], sizeof(uchar), 28, f);
    
    for(int i = 0; i < 29; ++i)
        for(int j = 0; j < 29; ++j)
            inputLayer[29 * i + j] = a[i][j] ? 0 : 1;
}

void makeStudyIteration(float *w[4], float *x[5], float &error)
{
    error = 0;
    for(int fileInd = 0; fileInd < 10; ++fileInd) // todo: write class Reader
    {
        FILE *f = fopen(dataFileName[fileInd], "rb");
        
        float y[10];
        for(int i = 0; i < 10; ++i)
            y[i] = i == fileInd ? 0.95 : -0.95;
        
        for(int imageInd = 0; imageInd < STUDYN; ++imageInd)
        {
            getInputLayer(x[0], f);
            
            calculate(w, x);
            
            error += getNetError(x[4], 10, y);
            
            backpropagate(w, x, y);
        }
        
        fclose(f);
    }
}

void initWeights(float *w, int n)
{
    for(int i = 0; i < n; ++i)
        w[i] = (rand() * 1. / RAND_MAX) * ((rand() & 1) ? 1 : -1) / 10;
}

void initArrays(float *w[4], float *x[5])
{
    w[0] = new float[156];
    w[1] = new float[7800];
    w[2] = new float[125100];
    w[3] = new float[1010];
    
    initWeights(w[0], 156);
    initWeights(w[1], 7800);
    initWeights(w[2], 125100);
    initWeights(w[3], 1010);
    
    x[0] = new float[841];
    x[1] = new float[1014];
    x[2] = new float[1250];
    x[3] = new float[100];
    x[4] = new float[10];
}

void study()
{
    float *w[4];
    float *x[5];
    
    initArrays(w, x);
    
    float lastError = 1e20;
    
    int step = 1;
    clock_t startTime = clock();
    
    for(;; ++step)
    {
        if(!(step % 60))
            ETA *= 0.3;
        
        float curError;
        
        makeStudyIteration(w, x, curError);
        
        if(fabs(curError - lastError) < EPS)
            break;
        
        lastError = curError;
        
        cerr << "study step #" << step << " : error = " << curError <<
        ", time = " << double(clock() - startTime) / CLOCKS_PER_SEC << "s" << endl;
        
        startTime = clock();
    }
    
    print("lw1.wei", w[0], 156);
    print("lw2.wei", w[1], 7800);
    print("lw3.wei", w[2], 125100);
    print("lw4.wei", w[3], 1010);
}

//---------------------------------------------------------------------------

int getResult(float output[10])
{
    float maxi = -1.0;
    int res = 0;
    for(int i = 0; i < 10; ++i)
    {
        if(maxi < output[i])
        {
            maxi = output[i];
            res = i;
        }
    }
    return res;
}

void test(const char *fileName, int expectedResult, int n)
{
	FILE *f = fopen(fileName, "rb");
	int rightResultCount = 0;
    
    float *w[4];
    w[0] = new float[156];
    w[1] = new float[7800];
    w[2] = new float[125100];
    w[3] = new float[1010];
    
    read("lw1.wei", w[0], 156);
    read("lw2.wei", w[1], 7800);
    read("lw3.wei", w[2], 125100);
    read("lw4.wei", w[3], 1010);
    
    float *x[5];
    x[0] = new float[841];
    x[1] = new float[1014];
    x[2] = new float[1250];
    x[3] = new float[100];
    x[4] = new float[10];
    
	for(int image = 0; image < n; ++image)
	{
        getInputLayer(x[0], f);
		calculate(w, x);
		int result = getResult(x[4]);
        
		if(result == expectedResult)
            ++rightResultCount;
    }
	cerr << fileName << " : " << rightResultCount << endl;
	fclose(f);
}

int main()
{
    cerr.precision(9);
	cerr << fixed;
    
    //-----------
    study();
    return 0;
    //------------
	
    clock_t t = clock();
	int n = 100;
    for(int i = 0; i < 10; ++i)
        test(dataFileName[i], i, n);
	cerr << double(clock() - t) / CLOCKS_PER_SEC / n / 10 << endl;
    
	return 0;
}
