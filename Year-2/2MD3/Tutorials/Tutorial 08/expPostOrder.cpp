#include <iostream>
#include <stack>
using namespace std;

bool isOperand(const char c){
    if((c >='a' && c<='z') || (c >='A' && c<='Z')){
        return true;
    }
    return false;
}

int priority(const char c){
    if(c == '+' || c =='-')
        return 1;
    if(c == '*' || c =='/')
        return 2;
    if(c == '^')
        return 3;
    return 0;
}

int  main() {
    stack<char> s;
    string exp = "A*(B+C)*D";
    for(int i = 0; i < exp.size(); ++i) {
        if (isOperand(exp[i])){
            cout << exp[i];
        }
        else if (exp[i] == '('){
            s.push(exp[i]);
        }
        else if (exp[i] != ')'){
            while (!s.empty()) {
	            char y = s.top(); s.pop();
                if (y != '(' && priority(y) >= priority(exp[i]))
                    cout << y;
                else {
                    s.push(y);
                    break;
                }
	        }
	        s.push(exp[i]);
        }
        else{  // exp[i] is ')'
            while (!s.empty()) {
	            char y = s.top(); s.pop();
                if (y != '(')
                    cout << y;
                else {
                    break;
                }
	        }
        }
    }
    while (!s.empty()) {
        char y = s.top(); s.pop();
        cout << y;
    }
    cout <<endl;
    return 0;
}

