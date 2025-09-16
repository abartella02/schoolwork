//Alex Bartella, 400308868
#include <iostream>
#include <math.h>
#include <string>
#include <string.h>
using namespace std;

class Term
{
    public:
        Term(int c=0.0, int e=0) : coef(c), exp(e) {}
        int coef;
        int exp;
};

typedef Term Elem;				// list element type Term
class DNode {					    // doubly linked list node
    private:
        Elem elem;					// node element value
        DNode* prev;				// previous node in list
        DNode* next;				// next node in list
        friend class DLinkedList;	// allow DLinkedList access
        friend class Polynomial;

        
};

class DLinkedList {				// doubly linked list
    public:
        DLinkedList();				// constructor
        ~DLinkedList();				// destructor
        bool empty() const;				// is list empty?
        const Elem& front() const;			// get front element
        const Elem& back() const;			// get back element
        void addFront(const Elem& e);		// add to front of list
        void addBack(const Elem& e);		// add to back of list
        void removeFront();				// remove from front
        void removeBack();				// remove from back

        friend class Polynomial;
    private:					// local type definitions
        DNode* header;				// list sentinels
        DNode* trailer;
    protected:					// local utilities
        void add(DNode* v, const Elem& e);		// insert new node before v
        void remove(DNode* v);			// remove node v
};

DLinkedList::DLinkedList() {			// constructor
    header = new DNode;				// create sentinels
    trailer = new DNode;
    header->next = trailer;			// have them point to each other
    trailer->prev = header;
}

bool DLinkedList::empty() const		// is list empty?
    { return (header->next == trailer); }

const Elem& DLinkedList::front() const	// get front element
    { return header->next->elem; }

const Elem& DLinkedList::back() const		// get back element
    { return trailer->prev->elem; }

DLinkedList::~DLinkedList() {			// destructor
    while (!empty()) removeFront();		// remove all but sentinels
    delete header;				// remove the sentinels
    delete trailer;
    //cout << "linked list destructor" << endl;
}

void DLinkedList::remove(DNode* v) {		// remove node v
    DNode* u = v->prev;				// predecessor
    DNode* w = v->next;				// successor
    u->next = w;				// unlink v from list
    w->prev = u;
    delete v;
}

void DLinkedList::removeFront()		// remove from font
    { remove(header->next); }
  
void DLinkedList::removeBack()		// remove from back
    { remove(trailer->prev); }

void DLinkedList::add(DNode* v, const Elem& e) { // insert new node before v
    DNode* u = new DNode;  
    u->elem = e;		// create a new node for e
    u->next = v;				// link u in between v
    u->prev = v->prev;				// ...and v->prev
    v->prev->next = u;
    v->prev = u;
  }

void DLinkedList::addFront(const Elem& e)	// add to front of list
    { add(header->next, e); }
  
void DLinkedList::addBack(const Elem& e)	// add to back of list
    { add(trailer, e); }

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

class Polynomial{
  public:
    Polynomial(); //constructor
    ~Polynomial(); //destructor 
    void insertTerm(const int c, const int e); //insert Term
    int eval(const int x); //evaluate polynomial at x
    friend ostream& operator<<(ostream& o, Polynomial& p); //overload operators
    Polynomial& operator+(const Polynomial &p2);
  private:
    string toString(); 
    DLinkedList* terms; //always sorted highest exp --> lowest exp
    int max, min; //store maximum and minimum value of exponents in polynomial
};

Polynomial::Polynomial(){
  terms = new DLinkedList; //dynamically allocate terms
}

void Polynomial::insertTerm(const int c, const int e){
  //cout << "Coef: " << to_string(c) << " Exp: " << to_string(e) << endl;
  DNode* curr = terms->header; //start at header
  int i = 0;
  max = 0;
  min = 0;
  do{ //find min and max of current list
    Elem elem = (curr->elem);
    if(i == 1){
      max = elem.exp;
      min = elem.exp;
    }else{
      if(elem.exp > max){
        max = elem.exp;
      }else if(elem.exp < min){
        min = elem.exp;
      }
    }
    i++;
    curr = curr->next;
  }while(curr != terms->trailer);

  int length = i; //find length of list for future use
    
    if((terms->empty()) || (e > max)){ //if list is empty or exponent is larger than the max
      terms->addFront(Term(c, e));
      max = e; //new max
      //cout << "~Added to FRONT " << endl;
    }else if(e < min){ //if exponent is smaller than the max
      terms->addBack(Term(c, e));
      min = e; //new min
      //cout << "~Added to BACK" << endl;
    }else{ //find term where new term needs to replace/be inserted in front of
      curr = terms->header->next;
      while(curr->elem.exp > e){
        curr = curr->next;
      }
  
      if(curr->elem.exp == e){ //if exponents are the same
        
        curr->elem.coef = c; //overwrite coefficient
      }else{
        terms->add(curr, Term(c, e)); //add in front of curr
      }
    }
}

string Polynomial::toString(){
  string poly = ""; //initialize string
  if(terms->empty()){
    return "0";
  }
  
  DNode* curr = terms->header->next; //start at first term element
  int c, e;
  do{
    c = curr->elem.coef;
    e = curr->elem.exp;

    if(c != 0){ //if coefficient is not zero, insert the term
      if(c != 1){ //if coefficient is 1 do not append coefficient or *
        poly += std::to_string(c); //concatenate coefficient
        poly += "*";
      }
      if(e == 1){ //if exponent is 1, print x without exponent
        poly += "x";
      }else if(e != 0){ //if exponent is not zero, print x with exponent
        poly += "x^"; 
        poly += std::to_string(e);
      }
      //poly += "*x^";
      
      
    }
    if(!(curr->next->elem.coef == 0)){ //if next term is 0 (will not be printed), do not append a "+"
        if(curr->next->elem.coef < 0){ //if next coefficient is negative, dont append "+" since a "-" will follow
          poly += " ";
        }else{ //if none of these conditions are met, append a "+"
          poly += " + ";
        }
      }
    curr = curr->next;
  }while(curr != terms->trailer); //repeat until end of the list
  return poly; //return complete string
}

istream& operator>>(istream& i, Polynomial& p){ //overloading >> operator
  int length; 
  int c; 
  int e; 
  int j = 0;
  
  cout << "Enter total number of terms: "; 
  i >> length;

  if(length != 0){
    while(j < length){
    cout << "Element #" << std::to_string(j+1) << ":" << endl;
    cout << "Enter coefficient: "; //prompt for coeff
    i >> c; //assign input to coefficient
    cout << "Enter exponent: "; //prompt for exp
    i >> e; //assign input to coeff
  
    p.insertTerm(c, e); //insert the term into the linked list
      j++;
    }
    cout << endl;
  }
  return i;
}

ostream& operator<<(ostream& o, Polynomial& p){ //overloading >> operator
  //cout << p.min;
  cout << p.toString(); //print using toString()

  return o;
}

int Polynomial::eval(const int x){
  int total = 0;
  DNode* curr = terms->header->next; //start at first term in polynomial
  while(curr != terms->trailer){
    total += (curr->elem.coef)*(pow( x, (curr->elem.exp) )); //evaluate the current term at x

    curr = curr->next; //iterate through the list and repeat
  }
  return total;
  
}

Polynomial& Polynomial::operator+(const Polynomial &p2){
  //Polynomial p = Polynomial();
  Polynomial* p = new Polynomial;
  //Polynomial p;
  
  int cond1 = 1, cond2 = 1, exp1, exp2; //cond1 and cond2 are used to evaluate whether or not the end of a list has been reached
  DNode* cur1 = terms->header->next;
  DNode* cur2 = p2.terms->header->next;
  
  while(cond1 || cond2){ //while one curr hasnt reached the end of the list...
    exp1 = cur1->elem.exp;
    exp2 = cur2->elem.exp;
    if(exp1 == exp2){ //if exponents of the term are equal
      p->insertTerm((cur1->elem.coef + cur2->elem.coef), exp1); //add coefficients
      cur1 = cur1->next; //iterate both terms
      cur2 = cur2->next;
    }else if(exp1 > exp2){ 
      p->insertTerm(cur1->elem.coef, exp1); //only insert exp1
      cur1 = cur1->next; //only iterate curr1
    }else if(exp2 > exp1){ //only insert exp2
      p->insertTerm(cur2->elem.coef, exp2); //only insert exp2
      cur2 = cur2->next;  //only iterate curr2
    }
    if(cur1 == terms->trailer){ //if curr reaches the end, set condition to zero
      cond1 = 0;
    }
    if(cur2 == p2.terms->trailer){
      cond2 = 0;
    }
    //if both terms reach the end, both conditions will be zero, and loop will break
  }
  return *p;
  //delete p;
}

Polynomial::~Polynomial(){
  //cout << toString() << " destructor" << endl;
  delete terms; //delete dynamically allocated
}


int main( ) {

    Polynomial p1;
    Polynomial p2;
    Polynomial p4;
    // cin >> pp;
    // cout << pp << endl;
    p1.insertTerm(-1,0);
    p1.insertTerm(1,0);
    p1.insertTerm(3,1);
    p1.insertTerm(5,2);
    cout << "p1 " << p1 << endl;

    p2.insertTerm(10,0);
    p2.insertTerm(-19,1);
    p2.insertTerm(10,3);
    cout << "p2 " << p2 << endl;

    //Polynomial p3;
    Polynomial p3;
    p3 = p1 + p2; //works
    Polynomial p5;
    p5 = p1 + p4; //works
    //cout << "test2" << endl;
    cout << "p3 " << p3 << endl;
    cout << "p4 " << p4 << endl;
    cout << "p5 " << p5 << endl;
    cin >> p4;
    cout << "p4 " << p4 << endl;
    return 0;
}