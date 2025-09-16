```//PL4 //PART 1
#include <pthread.h> //For thread creation and use of mutex locks
#include <stdio.h>
#include <stdlib.h>
#include<semaphore.h>


#define NUMWithdrawThreads 3
#define NUMDepositThreads 7
#define true 1

/* Code to declare mutex and Semaphores goes below*/
pthread_mutex_t mutex;
sem_t empty;
sem_t full;
/*Declare the thread functions*/
void *deposit(void *param);
void *withdraw(void *param);

/*Declare and initialize shared variable*/
int amount=0;

int main(int argc, char *argv[])
{
	int i,j;
    /*Declare Deposit and Withdraw threads*/
    pthread_t deposittid[NUMDepositThreads];
    pthread_t withdrawtid[NUMWithdrawThreads];
    pthread_attr_t attr;
    pthread_attr_init(&attr);

   if (argc != 3)
    {
        printf("Usage: %s <deposit> <withdraw>\n", argv[0]);
        return 1;
    }
   if(atoi(argv[1])<0)
   {
    printf("%d must be >=0\n",atoi(argv[1]));
   }

   // Initialize Mutex 
   if(pthread_mutex_init(&mutex,NULL)!=0);
   {
    printf("Error in initializing mutex \n");
   }

   //Create threads for deposit
   for(i=0;i<NUMDepositThreads;i++)
   {
    if(pthread_create(&deposittid[i],NULL, deposit, argv[1])!=0)
    {
        printf("Error in Deposit Thread %d\n",i);
         return 1;
    }
   
   }

   //Create threads for withdraw
   for(i=0;i<NUMWithdrawThreads;i++)
   {
    if(pthread_create(&withdrawtid[i],NULL, withdraw, argv[2])!=0)
    {
        printf("Error in withdraw Thread %d\n",i);
        return 1;
    }
   }

   //join threads
   for(i=0; i<NUMDepositThreads;i++)
   {
    pthread_join(deposittid[i], NULL);
    printf("Thread join failed\n");
    return 1;
   }

   for(i=0; i<NUMWithdrawThreads;i++)
   {
    pthread_join(withdrawtid[i], NULL);
    printf("Thread join failed\n");
    return 1;
   }
    pthread_mutex_destroy(&mutex);

    printf("Final amount= %d\n",amount);
	return 0;
}

void *deposit(void *param) {
    printf("Executing deposit function \n");
	while (true) { /* withdraw 50 */
        int v= atoi(param);
        pthread_mutex_lock(&mutex);
        amount=amount+v;
        printf("Amount after withdrawal=%d \n",amount);
        pthread_mutex_unlock(&mutex);
		pthread_exit(0);
        /*Enter code for the withdraw function. In here you need to use the mutex lock's pthread_mutex_lock() and pthread_mutex_unlock() functions  correctly.*/
	}
}

void *withdraw(void *param) {
    printf("Executing deposit function \n");
	while (true) { /* withdraw 50 */
        int v= atoi(param);
        pthread_mutex_lock(&mutex);

        if(amount >=v)
        {
            amount=amount-v;
        }
        printf("Amount after withdrawal=%d \n",amount);
        pthread_mutex_unlock(&mutex);
   		pthread_exit(0);
	}
}

```