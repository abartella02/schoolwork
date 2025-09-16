/*

QUESTION 1b

GROUP NUMBER: Group 29, L03 Section

NAME, MACID, STUDENTNUM: 
Alexander Bartella, bartella, 400308868
Jacqueline Leung, leungw18, 400314153

*/



#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <semaphore.h>
#include <unistd.h>


// declare threads
pthread_t thread1;
pthread_t thread2;
pthread_t thread3;

// declare thread runners
void *print(void *param);
void *sum(void *param);
void *doubles(void *param);

// declare semaphores for thread 1 and 2
sem_t sem1;
sem_t sem2;

// declare mutex lock
pthread_mutex_t using_data;

// create parameter structure
typedef struct {
    
    int arg1;   // first command line argument
    int arg2;   // second command line argument

    int retsum;     // result from sum runner

    int retdoubles;     // result from doubles runner

} parameters;



int main(int argc, char *argv[]) {

    parameters *arguments;

    arguments = (parameters *) malloc(sizeof(parameters));      // allocate memory


    // if there are too many arguments
    if(argc <= 2){
        printf("Not enough arguments.\n");
        return 0;
    }
    if (argc > 3) {
        printf("Too many arguments. \n");
        return 0;
    }

    // if the arguments are not less than 100
    if (atoi(argv[1])>=100 || atoi(argv[2])>=100 || atoi(argv[1])<0 || atoi(argv[2])<0 ) {
        printf("Arguments must be postive integers less than 100. \n");
        return 0;
    }


    arguments->arg1 = atoi(argv[1]);
    arguments->arg2 = atoi(argv[2]);

    // create threads
    pthread_create(&thread1, NULL, print, NULL);
    pthread_create(&thread2, NULL, sum, (void *) arguments);
    pthread_create(&thread3, NULL, doubles,(void *) arguments);

    // join threads
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    pthread_join(thread3, NULL);

    free(arguments);    // free memory

    printf("End of Main! \n");

    return 0;

}



void *print(void *param) {

    printf("I am the first thread! \n");

    sem_post(&sem1);    // tell thread2 this one is done

    pthread_exit(0);

}



void *sum(void *param) {

    parameters *arg = (parameters *)param;

    sem_wait(&sem1);    // wait for thread1

    pthread_mutex_lock(&using_data);

    arg->retsum = arg->arg1 + arg->arg2;

    printf("sum returns: %d \n", arg->retsum);

    pthread_mutex_unlock(&using_data);

    sem_post(&sem2);    // tell thread3 this one is done

    pthread_exit(0);

}



void *doubles(void *param) {

    parameters *arg = (parameters *)param;

    sem_wait(&sem2);    // wait for thread2

    pthread_mutex_lock(&using_data);

    arg->retdoubles = arg->retsum + arg->retsum;

    printf("doubles returns: %d \n", arg->retdoubles);

    pthread_mutex_unlock(&using_data);

    pthread_exit(0);

}



