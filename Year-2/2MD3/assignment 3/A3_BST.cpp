#include <list>
#include <iostream> 
#include <vector>
#include <queue>
#include <stack>
#include <math.h>
using namespace std;

class NonExistentElement{  // No such element Exception for BST
    public:
        NonExistentElement(const string& e)
            {errormsg = e;}
        string getMessage() const 
            {return errormsg;}
    private:
        string errormsg;
};

typedef int Elem;	// base element type

class KeyValues{
    private:
        int key;
        list<Elem> vals;
    public:
        int getKey(){
            return key;
        }
        list<Elem>& getValues(){
            return vals;
        }
        void setKey(int k){
            key = k;
        }
        void setValues(const list<Elem> &vals_lst){
            vals = vals_lst;
        }
        void addValue(const Elem v){
            vals.push_back(v);
        }
    friend class Position;

};

struct Node {					                            // a node of the tree
    KeyValues elt;					                        // element value 
    Node*   par;					                        // parent
    Node*   left;					                        // left child
    Node*   right;					                        // right child
    Node() : elt(), par(NULL), left(NULL), right(NULL) { }  // constructor
};


class Position {	 				                        // position in the tree
    private:
        Node* v;						                    // pointer to the node
    public:
        Position(Node* _v = NULL) : v(_v) { }		        // constructor
        KeyValues& operator*()					                // get element
            { return v->elt; }
        Position left() const				                // get left child
            { return Position(v->left); }
        Position right() const				                // get right child
            { return Position(v->right); }
        Position parent() const				                // get parent
            { return Position(v->par); }
        bool isRoot() const				                    // root of the tree?
            { return v->par == NULL; }
        bool isNull() const				                    // root of the tree?
            { return v == NULL; }
        bool isExternal() const				                // an external node?
            { return v->left == NULL && v->right == NULL; }
        bool operator==(const Position& p) const	        // are equal?
            { return v == p.v; }
        bool operator!=(const Position& p) const			// are not equal?
            { return v != p.v; }
        KeyValues& getElem()	const				                // get element
            { return v->elt; }
        list<Elem>& getList(){
            return (v->elt.getValues());
        } 
        //friend int SearchTree::countDescendants(Node* curr);
        //class SearchTree;       
        //friend class SearchTree;
    friend class BinaryTree;			                    // give tree access
};

typedef list<Position> PositionList;                        // list of positions

class BinaryTree {
    public:
        BinaryTree();					                    // constructor
        int size() const;					                // number of nodes
        bool empty() const;					                // is tree empty?
        Position root() const;				                // get the root
        PositionList positions() const;  			        // list of nodes
        void addRoot();					                    // add root to empty tree
        void expandExternal(const Position& p);		        // expand external node
        Position removeAboveExternal(const Position& p);	// remove p and parent
        // housekeeping functions omitted...
    protected: 						                        // local utilities
        void preorder(const Position& v, PositionList& pl) const;	// preorder utility
    private:
        Node* _root;					                    // pointer to the root
        int n;						                        // number of nodes
};

BinaryTree::BinaryTree()			                        // constructor
    : _root(NULL), n(0) { }

int BinaryTree::size() const			                    // number of nodes
    { return n; }

bool BinaryTree::empty() const			                    // is tree empty?
    { return size() == 0; }

Position BinaryTree::root() const                           // get the root
    { return Position(_root); }

void BinaryTree::addRoot()			                        // add root to empty tree
    { _root = new Node; n = 1; }

void BinaryTree::expandExternal(const Position& p) {        // expand external node
    Node* v = p.v;					                        // p's node
    v->left = new Node;					                    // add a new left child
    v->left->par = v;					                    // v is its parent
    v->right = new Node;				                    // and a new right child
    v->right->par = v;					                    // v is its parent
    n += 2;						                            // two more nodes
}

PositionList BinaryTree::positions() const {                // list of all nodes
    PositionList pl;
    preorder(root(), pl);					                // preorder traversal
    return PositionList(pl);				                // return resulting list
}

void BinaryTree::preorder(const Position& v, PositionList& pl) const {  // preorder traversal
    pl.push_back(Position(v));				                // add this node
    if (!v.left().isExternal())					            // traverse left subtree
        preorder(v.left(), pl);
    if (!v.right().isExternal())					        // traverse right subtree
        preorder(v.right(), pl);
}

Position BinaryTree::removeAboveExternal(const Position& p) {   // this is needed for removal
    Node* w = p.v;  Node* v = w->par;			                // get p's node and parent
    Node* sib = (w == v->left ?  v->right : v->left);
    if (v == _root) {					                        // child of root?
      _root = sib;					                            // ...make sibling root
        sib->par = NULL;
    }
    else {
      Node* gpar = v->par;				                        // w's grandparent
      if (v == gpar->left) gpar->left = sib; 		            // replace parent by sib
        else gpar->right = sib;
        sib->par = gpar;
    }
    delete w; delete v;					                        // delete removed nodes
    n -= 2;						                                // two fewer nodes
    return Position(sib);
}


class SearchTree {					                            // a binary search tree
    public: 						                            // public types
        class Iterator {	                      		        // an iterator/position
            private:
                Position v;						                // which entry
                Position treeRoot;
            public:
                Iterator(const Position& vv, const Position& r) : v(vv), treeRoot(r){ }		// constructor, root must be included
                KeyValues& operator*() { return v.getElem(); }			    // get entry
                bool operator==(const Iterator& p) const		// are iterators equal?
                    { return v == p.v; }
                bool operator!=(const Iterator& p) const		// are iterators not equal?
                    { return v != p.v; }
                Iterator& operator++();				            // inorder successor
                Iterator nextInPostorder() const;
                Iterator nextInLevelorder() const;
                Position getRoot() const{                       //root getter function
                    return treeRoot;
                }
                int depth(Position v) const{
                    int d;
                    Position vv = v;
                    while(!vv.isRoot()){
                        vv = vv.parent();
                        d++;
                    }
                    return d;
                }
            friend class SearchTree;				            // give search tree access
        };
    public:						                                // public functions
        SearchTree();					                        // constructor
        int size() const; 					                    // number of entries
        bool empty() const;					                    // is the tree empty?
        SearchTree::Iterator find(const Elem& k);				// find entry with key k
        SearchTree::Iterator insert(const Elem& k, const Elem& x);		// insert (k,x)
        void erase(const Elem& k); //throw(NonexistentElement);	// remove key k entry
        void erase( SearchTree::Iterator& p);			        // remove entry at p
        SearchTree::Iterator begin();					        // iterator to first entry
        SearchTree::Iterator end();					            // iterator to end entry
        PositionList getPositionList();                         // get the positions of the binary tree
        ///////////////////////////////////////////////////////////////////                      
        SearchTree::Iterator min();
        SearchTree::Iterator max();
        int rank(int k);
        SearchTree::Iterator floor(int k);
        SearchTree::Iterator ceil(int k);
        void eraseMin();
        void eraseMax();
        SearchTree::Iterator selectAtRank(int i);
        int countKeysBetween(int l, int h);
        list<SearchTree::Iterator> entriesBetween(int l, int h);
        int countDescendants(Position curr); //helper for rank
        SearchTree::Iterator median();
        int height();
        int depth(const SearchTree::Iterator& p);
        list<SearchTree::Iterator> inOrderBefore(int k);


    protected:						                            // local utilities
        Position root() const;					                // get virtual root
        Position finder(const Elem& k, const Position& v);		// find utility
        Position inserter(const Elem& k, const Elem& x);		// insert utility
        Position eraser(Position& w);				            // erase utility

    private: 						                            // member data
        BinaryTree T;					                        // the binary tree
        int n;
    //friend Iterator Iterator::getRoot();						                            // number of entries
};
			
SearchTree::SearchTree() : T(), n(0)                            // constructor
    { T.addRoot(); T.expandExternal(T.root()); }	            // create the super root

Position SearchTree::root() const                               // get virtual root
    { return T.root().left(); }				                    // left child of super root

int SearchTree::size() const                                    // get the size
    { return n; }

bool SearchTree::empty() const			                        // is search tree empty?
    { return size() == 0; }

SearchTree::Iterator SearchTree::begin() {                      // iterator to first entry
    Position v = root();					                    // start at virtual root
    while (!v.isExternal()) v = v.left();		                // find leftmost node
    return Iterator(v.parent(), root());
}

SearchTree::Iterator SearchTree::end()                          // iterator to end entry
    { return Iterator(T.root(), root()); }			                    // return the super root

PositionList SearchTree::getPositionList(){                     // ge the positions of the binary tree
    return T.positions();
}
				
SearchTree::Iterator& SearchTree::Iterator::operator++() {      // inorder successor
    Position w = v.right();
    if (!w.isExternal()) {				                        // have right subtree?
      do { v = w; w = w.left(); }			                    // move down left chain
        while (!w.isExternal());
    }
    else {
      w = v.parent();					                        // get parent
      while (v == w.right())				                    // move up right chain
        { v = w; w = w.parent(); }
      v = w;						                            // and first link to left
    }
    return *this;
}
	
Position SearchTree::finder(const Elem& k, const Position& v) { // find utility
    if (v.isExternal()) return v;			                    // key not found
    if (k < v.getElem().getKey()) return finder(k, v.left());	        // search left subtree
    else if (v.getElem().getKey() < k) return finder(k, v.right());	    // search right subtree
    else return v;					                            // found it here
}
			
SearchTree::Iterator SearchTree::find(const Elem& k) {          // find entry with key k
    Position v = finder(k, root());				                // search from virtual root
    if (!v.isExternal()) return Iterator(v, root());		            // found it
    else return end();					                        // didn't find it
}
					
Position SearchTree::inserter(const Elem& k, const Elem& x) {  // insert utility
    Position v = finder(k, root());				                // search from virtual root
    v = finder(k, v);
    if(v.isExternal()){
        T.expandExternal(v);
    }                       			                    // look further				                        // add new internal node
    //vector<Elem> vect = v.getList();
    v.getElem().setKey(k); 
    v.getList().push_back(x);                                                    // set the key!
    //vect.push_back(x);
    n++;						                                // one more entry
    return v;						                            // return insert position
}

SearchTree::Iterator SearchTree::insert(const Elem& k, const Elem& x)   // insert (k,x)
    { Position v = inserter(k, x); return Iterator(v, root()); }

Position SearchTree::eraser(Position& w) {                          // remove utility
    Position z;
    if (w.left().isExternal()) z = w.left();		                // remove from left case 1
    else if (w.right().isExternal()) z = w.right();	                // remove from right case 1
    else {						                                    // both internal? case 2
        z = w.right();					                            // go to right subtree
        do { z = z.left(); } while (!z.isExternal());	                // get leftmost node
        Position v = z.parent();
        //w->setKey(v->key()); w->setValue(v->value());	            // copy z's parent to w
        *w = v.getElem();	                                            // copy z's parent to v
    }
    n--;						                                    // one less entry
    return T.removeAboveExternal(z);			                    // remove z and its parent
}
					
void SearchTree::erase(const Elem& k){ //throw(NonexistentElement) { // remove key k entry
    Position w = finder(k, root());				                    // search from virtual root
    // if (v.isExternal())					                        // not found?
    //   throw NonexistentElement("Erase of nonexistent");
    eraser(w);						                                // remove it
}
	
void SearchTree::erase( Iterator& p)                                // erase entry at p
    { eraser(p.v); }
//////////////////////////////////////////////////////////////////////////////////

SearchTree::Iterator SearchTree::min(){ //works
    return begin();
}

void SearchTree::eraseMin(){ //works
    Position v = min().v;
    eraser(v);
}

SearchTree::Iterator SearchTree::max(){ //works
    Position pos = root();
    while(!pos.isExternal()){
        pos = pos.right();
    }
    return Iterator(pos.parent(), root());
}

void SearchTree::eraseMax(){ //works
    Position v = max().v;
    eraser(v);
}

int SearchTree::countDescendants(Position curr){ //needed for rank
    if(curr.isExternal()){                               
        return 0;
    }
    int left = 0, right = 0;
    Position l = curr.left();
    Position r = curr.right();
    if(!(l.isExternal())){
        left += countDescendants(l);
    }else{
        left = 0;
    }

    if(!(r.isExternal())){
        right += countDescendants(r);
    }else{
        right = 0;
    }

    return (1 + left + right);
}

int SearchTree::rank(int k){ //works
    Position curr = root();
    int count = 0;
    while(!curr.isExternal()){
        if(curr.getElem().getKey() == k){
            count += countDescendants(curr.left());
            return count;
        }
        if(curr.getElem().getKey() > k){
            curr = curr.left();
        }else if(curr.getElem().getKey() < k){
            count += countDescendants(curr.left())+1;
            curr = curr.right();
        }
    }
    throw NonExistentElement("Input key is not in tree");
}

SearchTree::Iterator SearchTree::floor(int k){ //works
    int j = 0;
    vector<Iterator> vect;
    for(Iterator i = begin() ; i != end() ; ++i){
        vect.push_back(i);
        if(k == (*i).getKey()){
            return i;
        }
        if((*i).getKey() > k){
            if(j > 0){return vect[j-1];}
            else{
                throw NonExistentElement("Input key is smaller than minimum key");
            }
        }
        j++;
    }
    return Iterator(max());
    
}

SearchTree::Iterator SearchTree::ceil(int k){ //works
    for(Iterator i = begin() ; i != end() ; ++i){
        if((*i).getKey()>=k){
            return i;
        }
    }
    throw NonExistentElement("Input key is larger than maximum key");
}

SearchTree::Iterator SearchTree::selectAtRank(int i){
    for(Iterator j = begin() ; j != end() ; ++j){
        if(rank((*j).getKey()) == i){
            return j;
        }
    }
    throw NonExistentElement("No node of rank \"i\" in tree");
}

int SearchTree::countKeysBetween(int l, int h){ //works
    int count = 0;
    for(Iterator j = begin() ; j != end() ; ++j){
        if((*j).getKey() >= l && (*j).getKey() <= h){
            count++;
        }
    }
    return count;
}

list<SearchTree::Iterator> SearchTree::entriesBetween(int l, int h){ //works
    list<Iterator> q;
    for(Iterator j = begin() ; j != end() ; ++j){
        if((*j).getKey() >= l && (*j).getKey() <= h){
            q.push_back(j);
        }
    }
    return q;
}

SearchTree::Iterator SearchTree::Iterator::nextInPostorder() const{ //works
    /*if(v == root()){}*/ //find a way for this to work
    if(v.parent().right() == v){ 
        return Iterator(v.parent(), getRoot());
    }else if(v.parent().left() == v){
        if(v.parent().right().isExternal()){  //no right node = go to parent
            if(v.parent().parent() == NULL){throw NonExistentElement("Specified node is the last in post-order");}
            return Iterator(v.parent(), getRoot());
        }
        Position vv = v.parent().right();
        while(!vv.isExternal()){
            if(!vv.left().isExternal()){
                vv = vv.left();
            }else if(!vv.right().isExternal()){
                vv = vv.right();
            }else{
                return Iterator(vv, getRoot());
            }
        }
        
    }
    throw NonExistentElement("Error: next node does not exist");

}
//queue
SearchTree::Iterator SearchTree::Iterator::nextInLevelorder() const{
    Position curr;
    queue<Position> q;
    bool cond = false;

    q.push(treeRoot);
    while (!q.empty()){
        curr = q.front();
        q.pop();
        if(!(curr.left().isExternal() || curr.left().isNull())){
            q.push(curr.left());
        }
        if(!(curr.right().isExternal() || curr.right().isNull())){
            q.push(curr.right());
        }
        if(curr == v){
            cond = true;
            break;
        }
    }
    if (cond != true || q.front().isNull()){
        throw NonExistentElement("Next Element in Level Order could not be Located");
    }
    
    return Iterator(q.front(), treeRoot);
}

list<SearchTree::Iterator> SearchTree::inOrderBefore(int k){
    Position curr = begin().v;
    stack<Position> s; 
    list<Position> l;

    while(!curr.left().isExternal()){
        
    }
    

}
//stack
/*
SearchTree::Iterator SearchTree::Iterator::nextInLevelorder() const{ //stack
    Position curr;
    stack<Position> q;

    q.push(treeRoot);
    while (!q.empty()){
        curr = q.top();
        q.pop();
        if(!curr.right().isExternal() && !curr.right().isNull()){
            q.push(curr.right());
        }
        if(!curr.left().isExternal() && !curr.right().isNull()){
            q.push(curr.left());
        }
        if(curr == v){
            break;
        }
    }
    return Iterator(q.top(), treeRoot);
}
*/
//queue++
/*
SearchTree::Iterator SearchTree::Iterator::nextInLevelorder() const{ //queue
    Position curr;
    queue<Position> q;
    vector<Position> levelOrder;

    q.push(treeRoot);
    while (!q.empty()){
        curr = q.front();
        levelOrder.push_back(curr);
        q.pop();
        if(!curr.left().isExternal()){
            q.push(curr.left());
        }
        if(!curr.right().isExternal()){
            q.push(curr.right());
        }
        /*
        if(curr == v){
            break;
        }
        
    }

    for (int i = 0; i < levelOrder.size(); i++)
    {
        cout << levelOrder[i].getElem().getKey() << ", ";
        
    }
    cout << endl;
    return Iterator(q.front(), treeRoot);
    
    //throw NonExistentElement("Next Node in level order does not exist");
}
*/


SearchTree::Iterator SearchTree::median(){ //works
    int med_rank = std::floor(rank((*max()).getKey())/2);
    return selectAtRank(med_rank);
}

/*
int SearchTree::heightHelp(Position& p, Position& target, int count){ //broken, doesnt work for nodes that are right children, might not need
    int rcount = 0, lcount = 0, cond = 0;
    if (p != target){
        if(!p.isExternal()){
            count++;
            if(p.left()!= NULL){
                lcount += count;
                Position pl = p.left();
                return heightHelp(pl, target, lcount);
            }
            if(p.right()!=NULL){
                --count;
                rcount += count;
                Position pr = p.right();
                return heightHelp(pr, target, rcount);
            }
        }
    }
    return count;
}
*/

int SearchTree::height(){ //works, but kinda jank
    int max = 0, d;
    for(Iterator i = begin() ; i != end() ; ++i){
        d = depth(i);
        if(d > max){max = d;}
    }
    return max;
}

int SearchTree::depth(const SearchTree::Iterator& p){ //works
    int count  = 0;
    Position pp = p.v;
    while(pp != root()){
        count++;
        pp = pp.parent();
    }

    return count;
}



int main(){
    //cout << " Hi " << endl;

    SearchTree t;
    t.insert(23, 8);
    t.insert(18, 8);
    t.insert(4, 8);
    t.insert(5, 8);
    t.insert(40, 8);
    t.insert(3, 8);
    t.insert(15, 8);
    //cout << "height node key: " << t.height() <<endl;
    cout << "Tree height (4): " << t.height() << endl;
    cout << "Median key: " << (*(t.median())).getKey() << endl;

    typedef SearchTree::Iterator It;
    vector<It> vect;
    for(It myit = t.begin() ; myit != t.end() ; ++myit){
        vect.push_back(myit);
    }
    int j = 2;
    cout << "depth of " << (*vect[j]).getKey() << " is " << t.depth(vect[j]) << endl;
    //cout << "next of " << (*vect[j]).getKey() << " is " << (*(vect[j].nextInLevelorder())).getKey() << endl;
    cout << endl;
    int i = 0;
    
    while(i<7){
        //cout << "Next of " << (*vect[i]).getKey() << " is " << (*(vect[i].nextInPostorder())).getKey() << endl;
        cout << "Rank of " << (*vect[i]).getKey() << " is " << t.rank((*vect[i]).getKey()) << endl;
        i++;
    }
    cout<<endl;
    i = 0;
    while(i<7){
        if((*vect[i]).getKey() != 15){
            cout << "Level Next of " << (*vect[i]).getKey() << " is " << (*(vect[i].nextInLevelorder())).getKey() << endl;
        }else{
            cout << "Level Next of 15 d.n.e (there would be an error thrown here)" << endl;
        }
        i++;
    }
    cout << endl;
    i = 0;
    while(i<7){
        if((*vect[i]).getKey() != 23){
        cout << "Next of " << (*vect[i]).getKey() << " is " << (*(vect[i].nextInPostorder())).getKey() << endl;
        }else{
            cout << "No next of " << (*vect[i]).getKey() << ", that's the root" << endl;
        }
        
        //cout << "Rank of " << (*vect[i]).getKey() << " is " << t.rank((*vect[i]).getKey()) << endl;
        i++;
    }

    
    //cout << "Next of " << *vect[0] << " is " << *vect[0].nextInPostorder() << endl;
    cout << endl;

    //cout << "Rank of 5: " << t.rank(5) << endl;
    cout << "Size: " << t.size() << endl;
    cout << "Ceil: " << (*(t.ceil(30))).getKey() << endl;
    cout << "---" << endl;
    t.erase(15);

    cout << "Post-Erase size: " << t.size() << endl;

    //cout << t.positions() << endl;

    //list<Position> res = t.positions();

    
    //typedef list<int>::iterator ITList;
    /*for(It myit = t.begin() ; myit != t.end() ; ++myit){
        cout << (*myit).getKey() << endl;
    }
    */

    cout << " -- -- -- -- " << endl;

    list<Position> a = t.getPositionList();
    //cout << a.front().getElem().getKey() << " -- " <<  a.size() << endl;

    cout << endl <<  " before erasing front and back" << endl;
    cout << "min: " << (*t.min()).getKey() << endl;
    cout << "max: " << (*t.max()).getKey() << endl;
    t.eraseMin(); t.eraseMax();

    cout << " after " << endl;
    cout << "min: " << (*t.min()).getKey() << endl;
    cout << "max: " << (*t.max()).getKey() << endl;

    //cout << *t.floor(2) << endl;
    cout << endl;
    return 0;
}