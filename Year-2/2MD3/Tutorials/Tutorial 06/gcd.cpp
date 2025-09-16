#include <iostream>
#include <math.h>
using namespace std;


int gcdIter(int m, int n){
    int gcd;
    if (n > m) {   // we have to swap them
        int temp = n;
        n = m;
        m = temp;
    }
    for (int i = 1; i <= n; i++) {  // start i from 1 to test if i is the common divisor and we check until b
        if (m % i == 0 && n % i ==0) {
            gcd = i;  // if i divides both a and b
        }
    }
    return gcd;

}

int gcdRec(int m, int n){  // Euclid's Algorithm
    //cout << "gcdRec(" << m << ", " << n << ")" << endl;
    if (n == 0)
        return m;
    return gcdRec(n, m % n);
}

int gcdRec2(int m, int n){  // Dijkstra's Algorithm
    //cout << "gcdRec2(" << m << ", " << n << ")" << endl;
    if (m == 0 || n == 0)
        return 0;
    else if (m == n)
        return m;
    else if (m > n)
        return gcdRec2(m-n, n);
    else 
        return gcdRec2(m, n-m);
}


int main( ) {
    cout << gcdRec2(539, 84) << endl;
    cout << "\n" << endl;

    cout << gcdRec(539, 84) << endl;
    cout << "\n" << endl;

    cout << gcdIter(539, 84) << endl;
}