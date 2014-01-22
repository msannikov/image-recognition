#include "util.h"

const float EPS = 3;
float ETA = 0.004;
//float ETA = 0.98;
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
    //return 1. / (1 + exp(-s));
}

float df(float s)
{
    return 1.7159 * 0.66666667 * (1. - sqr(0.66666667 * s));
    //return s * (1. - s);
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

void calcNeuronErrors(float *w[5], float *x[4], float *y, float *neuronError[5])
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
                w[0][ind] += -ETA * curError;
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
				w[1][ind] += -ETA * curError;
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
		w[2][ind] += -ETA * curError;
        ++ind;
        
		for(int i = 0; i < 1250; ++i, ++ind)
            w[2][ind] += -ETA * curError * neur[2][i];
	}
    
    for(int tmap = 0; tmap < 10; ++tmap)
	{
		int ind = tmap * 101;
        float curError = neuronError[4][tmap];
        
        w[3][ind] += -ETA * curError;
        ++ind;
        
		for(int i = 0; i < 100; ++i, ++ind)
            w[3][ind] += -ETA * curError * neur[3][i];
	}
}

void backpropagate(float *w[4], float *x[5], float *y, float *neuronError[5])
{
    memset(neuronError[0], 0, sizeof(float) * 841);
    memset(neuronError[1], 0, sizeof(float) * 1014);
    memset(neuronError[2], 0, sizeof(float) * 1250);
    memset(neuronError[3], 0, sizeof(float) * 100);
    memset(neuronError[4], 0, sizeof(float) * 10);
    
    calcNeuronErrors(w, x, y, neuronError);
    calcNewWeights(w, x, neuronError);
}

float getNetError(float *output, int size, float *y)
{
    float res = 0;
    for(int i = 0; i < size; ++i)
        res += sqr(output[i] - y[i]);
    return res / 2;
}

void makeTrainIteration(float *w[4], float *x[5], float &error, int n, float *neuronError[5])
{
    MNISTSampleReader reader("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
    
    for(int i = 0; i < n; ++i)
    {
        int expectedDigit = reader.getNextSample(x[0]);
        
        float y[10];
        for(int i = 0; i < 10; ++i)
            y[i] = i == expectedDigit ? 0.95 : -0.95;
            //y[i] = i == expectedDigit ? 0.98 : 0.02;

        calculate(w, x);
        
        error += getNetError(x[4], 10, y);
        
        backpropagate(w, x, y, neuronError);
    }
}

void initWeights(float *w, int n)
{
    for(int i = 0; i < n; ++i)
        w[i] = (rand() * 1. / RAND_MAX) * ((rand() & 1) ? 1 : -1) / 100;
}

void initArrays(float *w[4], float *x[5], bool training = 1, float *neuronError[5] = NULL)
{
    w[0] = new float[156];
    w[1] = new float[7800];
    w[2] = new float[125100];
    w[3] = new float[1010];
    
    x[0] = new float[841];
    x[1] = new float[1014];
    x[2] = new float[1250];
    x[3] = new float[100];
    x[4] = new float[10];
    
    if(training)
    {
        initWeights(w[0], 156);
        initWeights(w[1], 7800);
        initWeights(w[2], 125100);
        initWeights(w[3], 1010);
        
        neuronError[0] = new float[841];
        neuronError[1] = new float[1014];
        neuronError[2] = new float[1250];
        neuronError[3] = new float[100];
        neuronError[4] = new float[10];
    }
    else
    {
        read("weights/weight1", w[0], 156);
        read("weights/weight2", w[1], 7800);
        read("weights/weight3", w[2], 125100);
        read("weights/weight4", w[3], 1010);
    }
}

void printWeights(float *w[4])
{
    print("weights/weight1", w[0], 156);
    print("weights/weight2", w[1], 7800);
    print("weights/weight3", w[2], 125100);
    print("weights/weight4", w[3], 1010);
}

void freeMemory(float *w[4], float *x[5], float *neuronError[5])
{
    for(int i = 0; i < 5; ++i)
    {
        delete[] neuronError[i];
        delete[] x[i];
        if(i < 4)
            delete[] w[i];
    }
}

void train(int n)
{
    float *w[4];
    float *x[5];
    float *neuronError[5];
    
    initArrays(w, x, 1, neuronError);
    
    float lastError = 0;
    int step = 1;
    
    for(;; ++step)
    {
        clock_t startTime = clock();
        
        float curError = 0;
        
        makeTrainIteration(w, x, curError, n, neuronError);
        
        if(!(step % 50) && step)
        {
            printWeights(w);
            //ETA *= 0.5;
        }
        
        cerr << "training step #" << step << " : error = " << curError <<
            ", time = " << double(clock() - startTime) / CLOCKS_PER_SEC << "s" << endl;

        if(fabs(curError - lastError) < EPS || fabs(curError) < EPS)
            break;
        
        lastError = curError;
    }
    
    printWeights(w);
    
    freeMemory(w, x, neuronError);
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

void test(int n)
{
    MNISTSampleReader reader("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
    
	int rightResultCount = 0;
    
    float *w[4];
    float *x[5];

    initArrays(w, x, 0);

	for(int image = 0; image < n; ++image)
	{

        int expectedResult = reader.getNextSample(x[0]);

		calculate(w, x);
        
		int result = getResult(x[4]);
        
		if(result == expectedResult)
            ++rightResultCount;
    }
	cerr << "testing : " << rightResultCount * 1. / n << endl;
}

int main(int argc, char *argv[])
{
    cerr.precision(9);
	cerr << fixed;

    if(argc == 1)
    {
        cerr << "need one parameter: 1 - for training, 0 - for testing" << endl;
        return 0;
    }

    if(!atoi(argv[1]))
    {
        int n = 1000;
        test(n);
    }
    else
    {
        train(4000);
    }

	return 0;
}
