#include<iostream>
using namespace std;

typedef int Elem;
class SNode {					// circularly linked list node
    private:
        Elem elem;					// linked list element value
        SNode* next;				// next item in the list

    friend class SLinkedList;			// provide CircleList access
};

class SLinkedList {			// a singly linked list of Elems
    public:
        SLinkedList();			    // empty list constructor
        ~SLinkedList();			// destructor
        bool empty() const;				// is list empty?
        const Elem& front() const;		// get front element
        void addFront(const Elem& e);		// add to the front of list
        void removeFront();				// remove front item list
        
    private:
        SNode* head;				// pointer to the head of list
};

SLinkedList::SLinkedList()			// constructor
    : head(NULL) { }

SLinkedList::~SLinkedList()			// destructor
    { while (!empty()) removeFront(); }

bool SLinkedList::empty() const			// is list empty?
    { return head == NULL; }

const Elem& SLinkedList::front() const		// get front element
    { return head->elem; }

void SLinkedList::addFront(const Elem& e) {	// add to front of list
    SNode* v = new SNode;			// create new node
    v->elem = e;					// store data
    v->next = head;					// head now follows v
    head = v;						// v is now the head
}

void SLinkedList::removeFront() {		// remove front item
    SNode* old = head;				// save current head
    head = old->next;					// skip over old head
    delete old;						// delete the old head
}

int main( ) {
    SLinkedList a;				// list of integers
    a.addFront(1);
    a.addFront(2);
    a.addFront(3);
    a.addFront(4);

    return 0;
}