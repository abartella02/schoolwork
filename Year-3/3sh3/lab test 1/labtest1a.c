/*

QUESTION 1A

GROUP NUMBER: Group 29, L03

NAME, MACID, STUDENT NUMBER
Alexander Bartella, bartella, 400308868
Jacqueline Leung, leungw18, 400314153

*/


#include <unistd.h>
#include <stdio.h>
#include <sys/wait.h>


int main() {
    
    pid_t pid2, pid3, pid4, pid5, pid6;
	
    pid2 = fork(); //create p2 from p1
    if(pid2 > 0){ //p1
        pid3 = fork(); //create p3 from p1
        if(pid3 > 0){
            pid4 = fork(); //create p4 from p1
        }
        wait(NULL); //wait for p4
        wait(NULL); //wait for p3
        wait(NULL); //wait for p2
    }else if(pid2 == 0){ //p2
        pid5 = fork(); //create p5 from p2
        if(pid5 == 0){
            pid6 = fork(); //create p6 from p5
        }
        wait(NULL); //wait for p6/p5
    }
    printf("PID: %d, Parent PID: %d\n", getpid(), getppid()); //print


    return 0;
}
