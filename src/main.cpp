#include "util.h"

const float EPS = 3;
float ETA = 0.005; //обучаемость
//---------------------------------------------------------------------------

float f(float s)
{
	return 1.7159 * tanhf(0.66666667 * s);
}

float df(float s)
{
	return 1.7159 * 0.66666667 * (1. - sqr(0.66666667 * s));
}

void exec(float *input, float **w, float *bias, float *output, int inSize, int outSize)
{
	for(int i = 0; i < outSize; ++i)
	{
		float val = bias[i];

		for(int j = 0; j < inSize; ++j)
			val += input[j] * w[j][i];

		output[i] = f(val);
	}
}

void calculate(float **w[2], float *bias[2], float *x[3])
{
	exec(x[0], w[0], bias[0], x[1], 784, 50);
	exec(x[1], w[1], bias[1], x[2], 50, 10);
}
//----------------------------------------------------------------------------

void calcNeuronErrors(float **w[2], float *x[3], 
	float *y, float *neuronError[3])
{
	for(int i = 0; i < 10; ++i)
		neuronError[2][i] = df(x[2][i]) * (x[2][i] - y[i]);
	
	for(int i = 0; i < 50; ++i)
	{
		for(int j = 0; j < 10; ++j)
			neuronError[1][i] += w[1][i][j] * neuronError[2][j];
		neuronError[1][i] *= df(x[1][i]);
	}

	for(int i = 0; i < 784; ++i)
	{
		for(int j = 0; j < 50; ++j)
			neuronError[0][i] += w[0][i][j] * neuronError[1][j];
		neuronError[0][i] *= df(x[0][i]);
	}
}

void calcNewWeights(float **w[2], float *bias[2], float *neur[3], float *neuronError[3])
{
	for(int i = 0; i < 50; ++i)
		for(int j = 0; j < 10; ++j)
			w[1][i][j] += -ETA * neuronError[2][j] * neur[1][i];

	for(int i = 0; i < 784; ++i)
		for(int j = 0; j < 50; ++j)
			w[0][i][j] += -ETA * neuronError[1][j] * neur[0][i];		

	for(int i = 0; i < 10; ++i)
		bias[1][i] += -ETA * neuronError[2][i];

	for(int i = 0; i < 50; ++i)
		bias[0][i] += -ETA * neuronError[1][i];
}

void backpropagate(float **w[2], float *bias[2], float *x[3], 
	float *y, float *neuronError[3])
{
	memset(neuronError[0], 0, sizeof(float) * 784);
	memset(neuronError[1], 0, sizeof(float) * 50);
	memset(neuronError[2], 0, sizeof(float) * 10);
	
	calcNeuronErrors(w, x, y, neuronError);
	calcNewWeights(w, bias, x, neuronError);
}

float getNetError(float *output, int size, float *y)
{
	float error = 0;
	for(int i = 0; i < size; ++i)
		error += sqr(output[i] - y[i]);
	return error / 2;
}

void makeTrainIteration(float **w[2], float *bias[2], float *x[3], 
	float &error, int n, float *neuronError[3])
{
	MNISTSampleReader reader("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
	
	for(int i = 0; i < n; ++i)
	{
		int expectedDigit = reader.getNextSample(x[0]);
		
		float y[10];
		for(int j = 0; j < 10; ++j)
			y[j] = j == expectedDigit ? 0.95 : -0.95;

		calculate(w, bias, x);
		
		error += getNetError(x[2], 10, y);
		
		backpropagate(w, bias, x, y, neuronError);
	}
}

void initWeights(float *w, int n)
{
	for(int i = 0; i < n; ++i)
		w[i] = (rand() * 1. / RAND_MAX) * ((rand() & 1) ? 1 : -1) / 100;
}

void readWeights(float **w[2], float *bias[2])
{
	for(int i = 0; i < 784; ++i)
	{
		ostringstream fileName("w");
		fileName << "weights/w0" << i;
		read(fileName.str().c_str(), w[0][i], 50);
	}

	for(int i = 0; i < 50; ++i)
	{
		ostringstream fileName("w");
		fileName << "weights/w1" << i;
		read(fileName.str().c_str(), w[1][i], 10);
	}

	read("weights/b0", bias[0], 50);
	read("weights/b1", bias[1], 10);
}

void printWeights(float **w[2], float *bias[2])
{
	for(int i = 0; i < 784; ++i)
	{
		ostringstream fileName("w");
		fileName << "weights/w0" << i;
		print(fileName.str().c_str(), w[0][i], 50);
	}

	for(int i = 0; i < 50; ++i)
	{
		ostringstream fileName("w");
		fileName << "weights/w1" << i;
		print(fileName.str().c_str(), w[1][i], 10);
	}

	print("weights/b0", bias[0], 50);
	print("weights/b1", bias[1], 10);
}

void freeMemory(float **w[4], float*bias[2], float *x[3], float *neuronError[3] = NULL)
{
	for(int i = 0; i < 784; ++i)
		delete[] w[0][i];

	for(int i = 0; i < 50; ++i)
		delete[] w[1][i];

	delete[] w[0];
	delete[] w[1];

	delete[] bias[0];
	delete[] bias[1];

    for(int i = 0; i < 3; ++i)
    {
        if(neuronError)
			delete[] neuronError[i];
        delete[] x[i];
    }
}

void initArrays(float **w[2], float *bias[2], float *x[3], 
	int training = 1, float *neuronError[3] = NULL)
{
	// neurons
	x[0] = new float[784];
	x[1] = new float[50];
	x[2] = new float[10];

	// weights
	w[0] = new float*[784];
	w[1] = new float*[50];

	for(int i = 0; i < 784; ++i)
		w[0][i] = new float[50];

	for(int i = 0; i < 50; ++i)
		w[1][i] = new float[10];

	// biases
	bias[0] = new float[50];
	bias[1] = new float[10];

	if(training)
	{
		for(int i = 0; i < 784; ++i)
			initWeights(w[0][i], 50);
		
		for(int i = 0; i < 50; ++i)
			initWeights(w[1][i], 10);
		
		initWeights(bias[0], 50);
		initWeights(bias[1], 10);	

		//neuronErrors
		neuronError[0] = new float[784];
		neuronError[1] = new float[50];
		neuronError[2] = new float[10];
	}
	else
		readWeights(w, bias);
}

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
	
	float **w[2];
	float *bias[2];
	float *x[3];

	initArrays(w, bias, x, 0);

	for(int image = 0; image < n; ++image)
	{
		int expectedResult = reader.getNextSample(x[0]);

		calculate(w, bias, x);
		
		int result = getResult(x[2]);
		
		if(result == expectedResult)
			++rightResultCount;
	}

	cerr << "testing : " << rightResultCount * 1. / n << endl;

	freeMemory(w, bias, x);
}

void train(int n)
{
	float **w[2];
	float *bias[2];

	float *x[3];
	float *neuronError[3];
	
	initArrays(w, bias, x, 1, neuronError);
	
	float lastError = 0;
	int step = 1;

	for(;; ++step)
	{
		if(step > 30) break;

		clock_t startTime = clock();
		
		float curError = 0;
		
		makeTrainIteration(w, bias, x, curError, n, neuronError);
		
		printWeights(w, bias);

		cerr << "training step #" << step << " : error = " << curError <<
			", time = " << double(clock() - startTime) / CLOCKS_PER_SEC << "s" << "\n\t\t";

		test(n >> 2);

		if(fabs(curError - lastError) < EPS || fabs(curError) < EPS)
			break;
		
		lastError = curError;

		if(step && !(step % 15))
			ETA *= 0.7;
	}
	
	printWeights(w, bias);

	freeMemory(w, bias, x, neuronError);
}

//===============================================================
// Genetic algorithm
//===============================================================

const int MAXPOP = 150; // max size of population
const int chromSize = 39760;  /// !!!! remove all constants

struct chrom 
{
	const static float EPS = 10e-5;

	vector<float> genes;
	float fitness;  // == error while trainig
	float accuracy; // testing

	chrom()
	{
		genes.resize(chromSize);
	}

	bool operator == (chrom &c)
	{
		for (int i = 0; i < chromSize; i++)
			if (fabs(c.genes[i] - genes[i]) > EPS)
				return 0;
		return 1;
	}

	bool operator < (const chrom &c) const
	{
		return fitness < c.fitness;
	}
};

class GenAlgTrainer
{
	public:
		GenAlgTrainer(int sampleSize, bool readFromFile);

		int sampleSize;
		void train(int epochCount);
		void calcFitness(chrom &c);
		void crossover(chrom &c1, chrom &c2);
		float getRandVal();
		vector<chrom> chroms; // population of chromosoms
};

float GenAlgTrainer::getRandVal()
{
	return (rand() * 1. / RAND_MAX) * ((rand() & 1) ? 1 : -1) / 100;
}

GenAlgTrainer::GenAlgTrainer(int sampleSize, bool readFromFile)
{
	chroms.resize(MAXPOP);

	this->sampleSize = sampleSize;

	if(readFromFile)
	{
		float **w[2];
		float *bias[2];
		float *x[3];
		float *neuronError[3];

		initArrays(w, bias, x, 1, neuronError);
		initArrays(w, bias, x, 0);

		for(int chromInd = 0; chromInd < MAXPOP; ++chromInd)
		{
			int ind = 0;

			for(int i = 0; i < 784; ++i)
				for(int j = 0; j < 50; ++j)
					chroms[chromInd].genes[ind++] = w[0][i][j];

			for(int i = 0; i < 50; ++i)
				for(int j = 0; j < 10; ++j)
					chroms[chromInd].genes[ind++] = w[1][i][j];	

			for(int i = 0; i < 50; ++i)
				chroms[chromInd].genes[ind++] = bias[0][i];

			for(int i = 0; i < 10; ++i)
				chroms[chromInd].genes[ind++] = bias[1][i];	

			float curError;

			makeTrainIteration(w, bias, x, curError, sampleSize, neuronError);

			calcFitness(chroms[chromInd]); // refactor this
		}

		freeMemory(w, bias, x, neuronError);
	}
	else
	{
		for(int i = 0; i < MAXPOP; ++i)
		{
			for(int j = 0; j < chromSize; ++j)
				chroms[i].genes[j] = getRandVal();
			calcFitness(chroms[i]);
		}
	}
}

void GenAlgTrainer::crossover(chrom &c1, chrom &c2)
{
	int ind = rand() % (chromSize + 1);

	chrom r1, r2;

	for(int i = 0; i < chromSize; ++i)
	{
		r1.genes[i] = i < ind ? c1.genes[i] : c2.genes[i];
		r2.genes[i] = i >= ind ? c1.genes[i] : c2.genes[i];
	}

	if(r1 == r2)
	{
		int ind = rand() % chromSize;
		float val = rand() * 2. / RAND_MAX - 1;
		r1.genes[ind] = val;
	}

	calcFitness(r1);
	calcFitness(r2);

	chroms.pb(r1); 
	chroms.pb(r2);
} 

bool operator == (const chrom &l, const chrom &r)  /// why?
{
	for (int i = 0; i < chromSize; i++)
		if (fabs(l.genes[i] - r.genes[i]) > 10e-3)
			return 0;
	return 1; 
}

void GenAlgTrainer::train(int epochCount)
{
	for(int epoch = 0; epoch < epochCount; ++epoch)
	{
		clock_t startTime = clock();

		sort(all(chroms));
		chroms.erase(unique(all(chroms)), chroms.end());

		cerr << "epoch = " << epoch << endl;
		for(int i = 0; i < min(5, sz(chroms)); ++i)
			cerr << "error = " << chroms[i].fitness << "\t"
				<< "accuracy = " << chroms[i].accuracy << endl;
		
		int size = sz(chroms);
		for(int i = 0; i < size - 1; i += 2)
			crossover(chroms[i], chroms[i + 1]);

		for(int i = 0; i < MAXPOP; ++i)
		{
			int chromInd = rand() % sz(chroms);
			int geneInd = rand() % chromSize;
			float geneVal = rand() * 2. / RAND_MAX - 1;
			chroms[chromInd].genes[geneInd] = geneVal;
		}

		sort(all(chroms));
		chroms.erase(unique(all(chroms)), chroms.end());

		if(sz(chroms) > MAXPOP)
			chroms.resize(MAXPOP);

		cerr << "\ttime = " << double(clock() - startTime) / CLOCKS_PER_SEC << "s" << "\n";
		cerr << "------------------------" << endl;
	}
}

void GenAlgTrainer::calcFitness(chrom &c)
{
	float **w[2];
	float *bias[2];

	float *x[3];
	float *neuronError[3];

	initArrays(w, bias, x, 1, neuronError);

	int ind = 0;
	
	for(int i = 0; i < 784; ++i)
		for(int j = 0; j < 50; ++j)
			w[0][i][j] = c.genes[ind++];

	for(int i = 0; i < 50; ++i)
		for(int j = 0; j < 10; ++j)
			w[1][i][j] = c.genes[ind++];	

	for(int i = 0; i < 50; ++i)
		bias[0][i] = c.genes[ind++];

	for(int i = 0; i < 10; ++i)
		bias[1][i] = c.genes[ind++];

	MNISTSampleReader reader("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
	
	float error = 0;

	for(int i = 0; i < sampleSize; ++i)
	{
		int expectedDigit = reader.getNextSample(x[0]);
		
		float y[10];
		for(int j = 0; j < 10; ++j)
			y[j] = j == expectedDigit ? 0.95 : -0.95;

		calculate(w, bias, x);
		
		error += getNetError(x[2], 10, y);
	}

	c.fitness = error;
	//-----------------------

	MNISTSampleReader testReader = MNISTSampleReader("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
	
	int rightResultCount = 0;

	for(int image = 0; image < (sampleSize >> 2); ++image)
	{
		int expectedResult = testReader.getNextSample(x[0]);

		calculate(w, bias, x);
		
		int result = getResult(x[2]);

		if(result == expectedResult)
			++rightResultCount;
	}

	c.accuracy = rightResultCount * 1. / sampleSize * 4;

	freeMemory(w, bias, x, neuronError);
}

void mixedTraining(int sampleSize)
{
	// train(sampleSize);

	// GenAlgTrainer genAlgTrainer(sampleSize, 1);
	// genAlgTrainer.train(100000);

	GenAlgTrainer genAlgTrainer(sampleSize, 0);
	genAlgTrainer.train(100000);
}

int main(int argc, char *argv[])
{
	cerr.precision(9);
	cerr << fixed;

	mixedTraining(4000);
	// mixedTraining(100); 
	
	return 0;
}
