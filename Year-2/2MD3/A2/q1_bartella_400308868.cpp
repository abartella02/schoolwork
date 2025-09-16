//Alex Bartella, 400308868
#include <iostream>
#include <math.h>
using namespace std;

//////declarations
class Stats {
  public:
    Stats(int *arr, int len); //constructor
    virtual ~Stats(); //dont forget destructor
    double getMean();
    double getStdev();
    virtual void computeStats();
    virtual void printStats();

    //member vars
    int* array1;
    int length;
    int max;
  protected:
    double mean;
    double stdev;
};

class OneVarStats : public Stats{
  public:
    OneVarStats(int *arr, int len); //constructor
    void computeStats();
    void printStats();
    ~OneVarStats(); //destructor

    //member vars
    int histLength;
    int* hist; //arr
};

class TwoVarStats : public Stats{
  public: 
    TwoVarStats(int* arr1, int* arr2, int len); //constructor
    double computeCorrCoef();
    void computeStats();
    void printStats();
    ~TwoVarStats(); //destructor

    //member variables
    double correlationCoef;
    int* array2;
    int length;
};


//////member functions
//stats
Stats::Stats(int *arr, int len){
  array1 = new int[len]; //dynamically allocate new array

  for(int i = 0; i < len; i++){
    //deep copy of array
    array1[i] = arr[i];
    
    //finding maximum
    if(i == 0){
      max = array1[i];
    }else if(array1[i] > max){
      max = array1[i];
    }

  }
  length = len; //assigning length member var

}


double Stats::getMean(){ //getter function
  return mean;
}

double Stats::getStdev(){ //getter function
  return stdev;
}

void Stats::computeStats(){ //compute stats
  double sum = 0.0;
  for(int i = 0; i < length; i++){
    sum += array1[i]; //find sum of all elements in array
  }
  mean = sum/length; //store

  //std dev calculations
  double devsum = 0;
  for(int j = 0; j < length; j++){
    double xi = array1[j]; 
    devsum += pow((xi-mean), 2); //calculate sum from formula
  }
  double variance2 = devsum/length;  //variance^2
  stdev = sqrt(variance2); //store
}

void Stats::printStats(){
  cout << "mean= " << mean;
  cout << ", std= " << stdev << endl;
}

Stats::~Stats(){
  delete[] array1; //delete dynamically allocated array
}

//////////////////////////////////////////////////////////
//OneVarStats
OneVarStats::OneVarStats(int *arr, int len):Stats(arr, len){
  hist = new int[max - 1]; //dynamically allocate new histogram
  //cout << "MAX:  "<< max << endl;
  


}
void OneVarStats::computeStats(){
  double sum = 0;
  for(int i = 0; i < length; i++){
    sum += array1[i];
  }
  mean = sum/length; //store

  //std dev calculations
  double devsum = 0;
  for(int j = 0; j < length; j++){
    double xi = array1[j];
    devsum += pow((xi-mean), 2);
  }
  double variance2 = devsum/length;
  stdev = sqrt(variance2); //store

  //histogram calculations
  for(int i=0; i<=max; i++){ //for all numbers up until max in array
    //i = 1, 2, 3 ..

    int sum = 0;
    for(int j=0; j<length; j++){ 
      if(array1[j] == i){ //sum occurences of element i
        sum++;
      }
    }
    hist[i] = sum; //insert sum into term
  }
  histLength = max+1;
}

void OneVarStats::printStats(){ //print stats from stats and onevarstats
  cout << "mean= " << mean;
  cout << ", std= " << stdev << endl;
  for(int i=0; i<histLength; i++){
    cout << hist[i] << " "; //print the size of each bar in the histogram
  }
  cout << endl;
}

OneVarStats::~OneVarStats(){
  delete[] hist; //delete dynamically allocated histogram
}






//////////////////////////////////////////////////////////////////////////
TwoVarStats::TwoVarStats(int* arr1, int* arr2, int len):Stats(arr1, len){
  length = len; 
  //deep copy
  array2 = new int[length]; //dynamically allocate a new array
  for(int i = 0; i < len; i++){
    array2[i] = arr2[i]; //deep copy
  }
  
}

double TwoVarStats::computeCorrCoef() {  // works on array1 and array2 and count
    int count = length;
    double sumX = 0.0, sumY = 0.0, sumXY = 0.0;  // hold S(x), S(y), S(xy) respectively.
    double sumX2 = 0.0, sumY2 = 0.0;  // hold S(x^2), S(y^2) respectively.
    
    for (int i=0; i< count; i++){
        sumX += array1[i];
        sumY += array2[i];
        sumXY += array1[i] * array2[i];

        sumX2 += array1[i] *  array1[i];
        sumY2 += array2[i] *  array2[i];
    } 
    double corr_coef = (count * sumXY - sumX * sumY)/ (sqrt((count * sumX2 - sumX * sumX) * (count * sumY2 -  sumY * sumY))); 

    return corr_coef;
}

void TwoVarStats::computeStats(){ //same as stats class but with correlationCoef
  double sum = 0;
  for(int i = 0; i < length; i++){
    sum += array1[i];
  }
  mean = sum/length; //store

  //std dev calculations
  double devsum = 0;
  for(int j = 0; j < length; j++){
    double xi = array1[j];
    devsum += pow((xi-mean), 2);
  }
  double variance2 = devsum/length;
  stdev = sqrt(variance2); //store

  correlationCoef = computeCorrCoef();
}

void TwoVarStats::printStats(){ //print stats
  cout << "mean= " << mean;
  cout << ", std= " << stdev << endl;
  cout << "corr coef= " << correlationCoef << endl;
}
  
TwoVarStats::~TwoVarStats(){
  delete[] array2; //delete dynamically allocated array
}





int main( ) {
  int x[] = {2, 4, 7, 11, 5};
  int y[] = {5, 9, 14, 20, 10};
  int z[] = {14, 7, 4, 9, 21};
  
  int stats_len = 4;
  Stats* pp[stats_len];
  pp[0] = new Stats(x, 5);
  pp[1] = new OneVarStats (x, 5);
  pp[2] = new TwoVarStats (x, y, 5);
  pp[3] = new TwoVarStats (y, z, 5);
  
  for (int i=0; i < stats_len; i++){
    pp[i]->computeStats();
    cout << "\n";
  }
  for (int i=0; i < stats_len; i++){
    pp[i]->printStats();
    cout << "\n";
  }
}
