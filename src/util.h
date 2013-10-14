#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <ctime>
#include <iostream>

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