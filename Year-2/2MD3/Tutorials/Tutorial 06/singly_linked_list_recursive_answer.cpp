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
        void printList();              // prints the list
        int size();                    // return the size of the list
        void addTail(const Elem& e);  // add to the end of list
        void concatenate(SLinkedList& l2); //concats list l2 to the end of this
        void reverse();                 //reverses the list
        int sizeRec();
        Elem sumRec();
        void printRec();
        void printReverseRec();
        bool containsRec(const Elem& key);
        void addTailRec(const Elem& e);
        void reverseRec();
        
    private:
        SNode* head;				// pointer to the head of list
        int _sizeRec(SNode* node);
        Elem _sumRec(SNode* node);
        void _printRec(SNode* node);
        void _printReverseRec(SNode* node);
        bool _containsRec(SNode* node, const Elem& key);
        SNode* _addTailRec(SNode* node, const Elem& e);
        SNode* _reverseRec(SNode* node);
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

void SLinkedList::printList() {	// prints the list
    SNode* cur = head;             
    while(cur != NULL) {
        cout << cur->elem << "\t" ;     // we print in one line
        cur = cur->next;
    }
    cout << endl;       // and put an endl when all nodes were printed
}

int SLinkedList::size() {	// remove front item
    int counter = 0;        // we count the nodes
    SNode* cur = head;  // we start from the head
    while(cur != NULL) {  // until we reach to the null at the end
        counter += 1;
        cur = cur->next;
    }
    return counter;
}

void SLinkedList::addTail(const Elem& e) {	// add to the end of list
    SNode* v = new SNode;			// create new node
    v->elem = e;					        // store data
    v->next = NULL;	// since it will be the new last node, it has to point to Null.
    SNode* cur = head;     // use cur to traverse the list
    while(cur->next != NULL) {  // continue until the last node
        cur = cur->next;
    } // when exiting this loop, cur will be the last node.
    cur->next = v;	// we make a link from the last node to the new last node
}

void SLinkedList::concatenate(SLinkedList& l2) {	// remove front item
    SNode* cur = head;
    while(cur->next != NULL) {      // this loop find the last node
        //cout << tmp->elem << endl;
        cur = cur->next;
    }                   // cur will be the last node when going out of this  loop
    cur->next = l2.head;  // set the next pointer of cur to the head of l2
    l2.head = NULL;        // logically make l2 empty (remove it)
}

void SLinkedList::reverse() {
    // pvr is Null and this will be set to first node's next pointer at the first iteration
    SNode* prv = NULL;   // node before the current node
    SNode* cur = head;   // current node
    SNode* nxt = NULL;   // node after the current node
    while(cur->next != NULL) {  // continue until you reach last node
        nxt = cur->next;        // save next pointer of current before changing it
        cur->next = prv;        // update current's next
        prv = cur;              // update prv before advancing cur (we save cur here)
        cur = nxt;              // update  cur at the end
    } // when going out of this loop, cur is the last node of the original list  
    cur->next = prv;  // set cur's next
    head = cur;        // set new head
}

int SLinkedList::_sizeRec(SNode* node){	// remove front item
    if (node == NULL)
        return 0;
    else
        return 1 + _sizeRec(node->next);
}

int SLinkedList::sizeRec(){
    return _sizeRec(head);
}

Elem SLinkedList::_sumRec(SNode* node){
    if (node == NULL)
        return 0;
    else
        return node->elem + _sumRec(node->next);
}

Elem SLinkedList::sumRec(){
    return _sumRec(head);
}

void SLinkedList::_printRec(SNode* node) {
    if (node == NULL)
        return;
    else {
        cout << node->elem << "\t";
        _printRec(node->next);
    }
}

void SLinkedList::printRec(){
    return _printRec(head);
}

void SLinkedList::_printReverseRec(SNode* node) {
    if (node == NULL)
        return;
    else {
        _printReverseRec(node->next);
        cout << node->elem << "\t";
    }
}

void SLinkedList::printReverseRec(){
    return _printReverseRec(head);
}

bool SLinkedList::_containsRec(SNode* node, const Elem& key) {
    if (node == NULL)
        return false;
    else {
        if (node->elem == key)
            return true;
        else
            return _containsRec(node->next, key);
    }
}

bool SLinkedList::containsRec(const Elem& key){
    return _containsRec(head, key);
}

SNode* SLinkedList::_addTailRec(SNode* node, const Elem& e) {	// add to front of list
    if (node == NULL){
        SNode* v = new SNode;			// create new node
        v->elem = e;					// store data
        v->next = NULL;					// head now follows v
        return v;
    }
    else{
        node->next = _addTailRec(node->next, e);
        return node;
    }
}

void SLinkedList::addTailRec(const Elem& e) {	// add to front of list
    head = _addTailRec(head, e);
}

SNode* SLinkedList::_reverseRec(SNode* node) {
    if (node == NULL || node->next == NULL)
        return node;
    else {
        SNode* n = _reverseRec(node->next);
        node->next->next = node;
        node->next = NULL;
        return n;
    }
}

void SLinkedList::reverseRec(){
    head = _reverseRec(head);
}

int main( ) {
    SLinkedList a;				// list of integers
    a.printRec();   cout << "\n";  // print empty list
    a.addFront(1);
    a.addFront(2);
    a.addFront(3);
    a.addFront(4);
    //cout << a.front() << endl;
    a.printList();
    cout << a.size() << endl;

    cout << " Rec " << a.sizeRec() << endl;
    cout << " Rec " << a.sumRec() << endl;
    a.printRec();        cout << "\n";
    a.printReverseRec(); cout << "\n";
    cout << " Rec " << a.containsRec(2) << endl;
    cout << " Rec " << a.containsRec(8) << endl;
    cout << " Rec " << a.containsRec(4) << endl;
    a.addTailRec(5);
    a.printRec(); cout << "\n";

    a.printList();

    SLinkedList b;
    b.addTailRec(10);
    b.addTailRec(11);
    b.addTailRec(12);
    b.addTailRec(13);

    b.printRec(); cout << "\n";

    b.printList();

    b.reverseRec();

    b.printRec(); cout << "\n";

    b.printList();
}