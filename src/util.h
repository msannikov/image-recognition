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

class MNISTSampleReader
{
    FILE *imagesFile;
    FILE *labelsFile;
    
public:
    ~MNISTSampleReader()
    {
        fclose(imagesFile);
        fclose(labelsFile);
    }
    
    MNISTSampleReader(const char *imagesFileName, const char *labelsFileName)
    {
        imagesFile = fopen(imagesFileName, "rb");
        labelsFile = fopen(labelsFileName, "rb");
        
        fseek(labelsFile, 8, SEEK_SET);
        fseek(imagesFile, 16, SEEK_SET);
    }
    
    int getNextSample(float *inputLayer)
    {
        uchar a[29][29];
        memset(a, 0, sizeof a);
        for(int i = 0; i < 28; ++i)
            fread(&a[i], sizeof(uchar), 28, imagesFile);
        
        for(int i = 0; i < 29; ++i)
            for(int j = 0; j < 29; ++j)
                inputLayer[29 * i + j] = a[i][j] ? 0 : 1;
        
        uchar label;
        fread(&label, sizeof(uchar), 1, labelsFile);
        
        return label;
    }
};
