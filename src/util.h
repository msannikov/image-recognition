#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

template<class T> T sqr(T a) { return a * a; }

typedef unsigned char uchar;
using namespace std;

void print(const char *fileName, float *a, int n)
{
	FILE *f = fopen(fileName, "wb");
	fwrite(a, sizeof(float), n, f);
	fclose(f);
}

void read(const char *fileName, float *a, int n)
{
	FILE *f = fopen(fileName, "rb");
	fread(a, sizeof(float), n, f);
	fclose(f);
}

class SampleReader
{
    FILE **f;
    
    int k;
    vector<int> sequence;
    int curInd;
    
public:
    ~SampleReader()
    {
        for(int i = 0; i < k; ++i)
            fclose(f[i]);
    }

    SampleReader(int n, const char **fileNames, int k)
    {
        this->k = k;
        f = new FILE*[k];
        for(int i = 0; i < k; ++i)
            f[i] = fopen(fileNames[i], "rb");
        sequence.resize(k * n);
        for(int i = 0; i < k; ++i)
        {
            for(int j = 0; j < n; ++j)
                sequence[i * n + j] = i;
        }
        srand(4231121);
        random_shuffle(sequence.begin(), sequence.end());
        
        curInd = 0;
    }
    
    int getNextSample(float *inputLayer)
    {
        int curFileInd = sequence[curInd++];
        
        uchar a[29][29];
        memset(a, 0, sizeof a);
        for(int i = 0; i < 28; ++i)
            fread(&a[i], sizeof(uchar), 28, f[curFileInd]);
        
        for(int i = 0; i < 29; ++i)
            for(int j = 0; j < 29; ++j)
                inputLayer[29 * i + j] = a[i][j] ? 0 : 1;

        return curFileInd;
    }
};