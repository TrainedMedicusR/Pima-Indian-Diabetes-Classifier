#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>
#include <math.h>
#include<queue>
using namespace std;

typedef pair<double,double> meanVar;
typedef vector<double> features;
typedef pair<features,int> scoreClass;
typedef pair<double,int> distanceScore;

class Classifer {
public:
  void LoadTrainging(vector<scoreClass> vClass) {
    for(int i = 0; i < vClass.size(); ++i) ProcessClass(vClass[i]);
  }
  //Preprocess when each class is read
  virtual void ProcessClass(scoreClass sc) {};
  //Train data
  virtual void Train() = 0;
  //Return a list of prediction
  virtual vector<int> Predict(vector<scoreClass> tSet) =0;
  //Reset the training
  virtual void Clear() = 0;
protected:
  int _trainLength;
  int _attrLength;
};

class NBClassifer: public Classifer{
public:
  virtual void Train() {
    for(int i = 0; i < _attrLength; ++i) {
        __caculateMeanVariance(i,_yesParams[i],_yesGroup);
        __caculateMeanVariance(i,_noParams[i],_noGroup);
    }
  }
  virtual vector<int> Predict(vector<scoreClass> tSet) {
    double yesRate= (double)_yesGroup.size()/ static_cast<double>(_yesGroup.size() + _noGroup.size());
    double noRat = (double)_noGroup.size()/ static_cast<double>(_yesGroup.size() + _noGroup.size()) ;
    vector<int> ans;
    for(int i = 0; i < tSet.size(); ++i) {
      double pYes = __caculateRate(tSet[i].first,_yesParams, yesRate);
      double pNo =__caculateRate(tSet[i].first,_noParams, noRat);
      if(pYes < pNo) {
        cout <<"no"<<endl;
        ans.push_back(-1);
      } else {
        cout <<"yes"<<endl;
        ans.push_back(1);
      }
    }
    return ans;
  }
  virtual void ProcessClass(scoreClass sc)  {
    if(_yesGroup.size() == 0 && _noGroup.size() == 0) {
      _attrLength = sc.first.size();
      _yesParams.assign(_attrLength, meanVar(0,0)); _noParams.assign(_attrLength,meanVar(0,0));
    }
    vector<features>* cGroup = nullptr;
    vector<meanVar>* cParams = nullptr;
    if(sc.second == 1) {
      cGroup = &_yesGroup;
      cParams = &_yesParams;
    } else {
      cGroup = &_noGroup;
      cParams = &_noParams;
    }
    cGroup->push_back(sc.first);
    for(int i = 0; i < _attrLength; ++i)
      (*cParams)[i].first += sc.first[i];
  }
  virtual void Clear() {
    _yesGroup = vector<features>();
    _noGroup = vector<features> ();
    _yesParams = vector<meanVar> ();
    _noParams = vector<meanVar> ();
  }
private:
  vector<features> _yesGroup;
  vector<features> _noGroup;
  vector<meanVar> _yesParams;
  vector<meanVar> _noParams;
  void __caculateMeanVariance(int attrI, meanVar& mv, vector< features >& group) {
    int n = group.size();
    if(n == 1) {
      mv =meanVar(-1,-1); return;
    } else {
      mv.first/= n;
      for(int i = 0; i< n; ++i) mv.second += pow((group[i][attrI] - mv.first),2.0);
      mv.second = sqrt(mv.second/(n-1));
    }
  }
  double __caculateRate(features& para, vector<meanVar>& params, double pRate) {
    double odds = 1;
    for(int i = 0; i < _attrLength; ++i) {
        odds *= __probDens(para[i],params[i]);
    }
    return odds * pRate;
  }
  double __probDens(double mu, meanVar menVar) {
    double  diff = mu-menVar.first;
    if(menVar.second==0) return 1;
    double ans =  exp (-0.5  * pow((diff/menVar.second),2.0))/(menVar.second * sqrt(2.0*3.14));
    return ans;
  }
};

class KNearestClassifer: public Classifer{
public:
  KNearestClassifer(int k): _k(k) {}
  virtual void Train() {
    _trainLength = predictList.size();
  }
  virtual vector<int> Predict(vector<scoreClass> tSet) {
    vector<int> ans; double cost;
    for(int t = 0; t < tSet.size(); ++t) {
      priority_queue<distanceScore> sd;
      for(int i = 0; i < _trainLength; ++i)  {
        cost = __caculateRate(tSet[t].first,predictList[i].first);
        if(sd.size() < _k) {
          sd.push(distanceScore(cost,predictList[i].second));
        } else if(sd.size() >= _k && (sd.top().first > cost || (sd.top().first == cost && predictList[i].second == 1))) {
            sd.pop();
            sd.push(distanceScore(cost,predictList[i].second));
        }
      }
      int cnt = 0;
      while(!sd.empty()) {
        cnt += sd.top().second;
        sd.pop();
      }
      if(cnt >= 0)
        cout <<"yes"<<endl;
      else
        cout <<"no"<<endl;
      ans.push_back(cnt/abs(cnt));

    }
    return ans;
  }
  virtual void ProcessClass(scoreClass sc)  {
    predictList.push_back(sc);
  }
  virtual void Clear() {
    predictList = vector<scoreClass>();
  }
private:
  int _k;
  vector<scoreClass> predictList;
  double __caculateRate(features& para, features& params) {
    if(para.size() != params.size()) {
      cout <<"Incorrect Number of Input for tesrt data "<<endl;
      exit(1);
    }
    double sol = 0;
    for(int i = 0; i < para.size(); ++i) sol += pow(para[i] - params[i],2);
    return sqrt(sol);
  }
};

vector<scoreClass> trainingSet;
vector<scoreClass> testSet;

void LoadDate(string input_FileName, vector<scoreClass>& mySet) {
  string tmp, seg;
  fstream f;
  f.open(input_FileName, ios::in);
  while (f >> tmp) {
     stringstream s(tmp);
     scoreClass sc(features(),0);
     while (getline(s, seg, ',')) {
       if(seg == "yes" || seg == "no")
        sc.second = (seg == "yes"? 1:-1);
      else
        sc.first.push_back(stod(seg));
     }
     mySet.push_back(sc);
  }
  f.close();
}
void WriteToFile(vector< vector<scoreClass> >& vScore,int n){
  ofstream f;
  f.open ("pima-folds.csv");
  int m  = vScore[0][0].first.size();
  for(int i = 0; i < n; ++i) {
    f<<"fold"<<to_string(i+1)<<"\n";
    for(int j = 0; j < vScore[i].size(); ++j) {
      string s = "";
      features params = vScore[i][j].first;
      for(int k = 0; k <params.size(); ++k)
        s += to_string(params[k]) + ",";
      s += (vScore[i][j].second == 1? "yes":"no") + string("\n");
      f << s;
    }
    if(i != n-1)
      f <<"\n";
  }
  f.close();
}
void PreProcess(vector< vector<scoreClass> >& vScore, int n) {
  int m = trainingSet.size();
  vector<scoreClass> postiveClass, negativeClass;
  for(int i = 0; i < m; ++i) {
      if(trainingSet[i].second == 1)
        postiveClass.push_back(trainingSet[i]);
      else
        negativeClass.push_back(trainingSet[i]);
  }
  int postiveBatch = postiveClass.size()/n, negativeBatch = negativeClass.size()/n;
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < postiveBatch; ++j) {
      vScore[i].push_back(postiveClass.back());
      postiveClass.pop_back();
    }
    for(int j = 0; j < negativeBatch; ++j) {
      vScore[i].push_back(negativeClass.back());
      negativeClass.pop_back();
    }
  }
  for(int i = 0; i < n; ++i) {
    if(!postiveClass.empty()) {
      vScore[i].push_back(postiveClass.back());
      postiveClass.pop_back();
    } else if(!negativeClass.empty()){
      vScore[i].push_back(negativeClass.back());
      negativeClass.pop_back();
    }
  }
}

void PeformNFold(int n,Classifer* myClassifer) {
  int cnt = 0;
  features accuracy (n,0);
  vector<int> output;
  vector< vector<scoreClass> > vScore(n,vector<scoreClass> ());
  PreProcess(vScore, n);
  WriteToFile(vScore,n);
  for(int i = 0; i < n; ++i) {
    myClassifer->Clear();
    for(int j = 0; j < n; ++j)
      if(j != i) myClassifer->LoadTrainging(vScore[j]);
    myClassifer->Train();
    output = myClassifer->Predict(vScore[i]);
    cnt=0;
    for(int j = 0; j < output.size(); ++j) if(output[j] == vScore[i][j].second) ++cnt;
    accuracy[i] = static_cast<double>(cnt)/output.size();
  }
  for(int i = 0; i < n; ++i)
    cout << "Nfold" << i<<": "<<accuracy[i]<<"   "<<vScore[i].size()<< endl;
}
int main(int argc, char* argv[]) {
    if(argc != 4) {
      cout <<"Incorrect number of input"<<endl;
      return 1;
    }
    string trainFilePath = argv[1], testFilePath = argv[2], algo = argv[3], tmp, seg;
    Classifer* myClassifer = nullptr;
    if(algo == "NB"){
      myClassifer = new NBClassifer();
    } else if(algo.length() >= 3&& algo.substr(algo.length()-2) == "NN") {
      myClassifer = new KNearestClassifer(stoi(algo.substr(0,algo.length()-2)));
    } else {
      cout <<"Reading algorithm error: Please enter a correct algorithm." <<endl;
      return -1;
    }
    LoadDate(trainFilePath, trainingSet);
    LoadDate(testFilePath,testSet);
    //PeformNFold(10,myClassifer);
    myClassifer->LoadTrainging(trainingSet);
    myClassifer->Train();
    myClassifer->Predict(testSet);
}
