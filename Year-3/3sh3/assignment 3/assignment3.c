#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/*
ASSIGNMENT 3
    ALex Bartella, 400308868
    Jacqueline Leung, 400314153
*/


/*
REFERENCES USED

Geeks for Geeks - "FCFS Disk Scheduling Algorithms"
https://www.geeksforgeeks.org/fcfs-disk-scheduling-algorithms/
    Used as a reference for the FCFS function

Geeks for Geeks - "Program for SSTF disk scheduling algorithm"
https://www.geeksforgeeks.org/program-for-sstf-disk-scheduling-algorithm/
    Used as a reference for the SSTF function

Geeks for Geeks - "Bubble Sort Algorithm"
https://www.geeksforgeeks.org/bubble-sort/
    Used as a sorting algorithm in SSTF

Geeks for Geeks - "Disk Scheduling Algorithms"
https://www.geeksforgeeks.org/disk-scheduling-algorithms/
    Used as a reference for all of the functions



*/




#define REQUESTS 20


int requests_array[REQUESTS];
int head_position;
int direction;      // 0 is left and 1 is right


// FCFS function
int FCFS(int head_position, int requests_array[]) {

    int head_movements = 0;

    int difference, current_number;

    for (int i=0; i<REQUESTS; i++) {
        if (i == REQUESTS - 1) {
            printf("%d \n", requests_array[i]);
            break;
        }
        printf("%d ", requests_array[i]);
    }

    for (int i=0; i<REQUESTS; i++) {

        current_number = requests_array[i];
        difference = abs(head_position - current_number);
        head_movements += difference;

        head_position = current_number;      

    }

    return head_movements;

}



// SSTF function
int SSTF(int head_position, int requests_array[], int sorted_array[]) {


    int index_lessthan = 0;
    int index_greaterthan = 0;
    int init_head_position = head_position;

    int SSTF_lessthan[REQUESTS];
    int SSTF_greaterthan[REQUESTS];

    int head_in_requests;       // 0 is in, 1 is not in
    for (int i=0; i<REQUESTS; i++) {
        if (head_position == requests_array[i]) {
            head_in_requests = 0;
            break;
        }
        head_in_requests = 1;
    }

    // adjust "head position" (first request to be fulfilled in this case), if the given head position is not in array of requests
    if (head_in_requests == 1) {
        int min = head_position;
        int min_index, diff;
        for(int i=0; i<REQUESTS; i++){
            diff = head_position - requests_array[i];
            if (diff < min && diff > 0) {
                min = diff;
                min_index = i;
            }
        }
        head_position = requests_array[min_index];
    }


    // found amount of numbers lower/higher than head
    for (int i=0; i<REQUESTS; i++) {
        int difference = head_position - requests_array[i];

        if (difference > 0) {
            SSTF_lessthan[index_lessthan] = requests_array[i];
            index_lessthan += 1;
        }
        if (difference < 0) {
            SSTF_greaterthan[index_greaterthan] = requests_array[i];
            index_greaterthan += 1;

        }
    }

    // sort the numbers lower than head
    for (int i=0; i<index_lessthan; i++) {
        for (int j=i+1; j<index_lessthan; j++) {
            if (SSTF_lessthan[i] < SSTF_lessthan[j]) {
                int temp = SSTF_lessthan[i];
                SSTF_lessthan[i] = SSTF_lessthan[j];
                SSTF_lessthan[j] =temp;
             }
        }
    }

    // sort the numbers higher than head
    for (int i=0; i<index_greaterthan; i++) {
        for (int j=i+1; j<index_greaterthan; j++) {
            if (SSTF_greaterthan[i] > SSTF_greaterthan[j]) {
                int temp = SSTF_greaterthan[i];
                SSTF_greaterthan[i] = SSTF_greaterthan[j];
                SSTF_greaterthan[j] =temp;
             }
        }
    }

    // merge the lower and higher arrays
    int SSTF_order[REQUESTS];
    SSTF_order[0] = head_position;

    for (int i=1; i<=index_lessthan; i++) {
        SSTF_order[i] = SSTF_lessthan[i-1];
    }

    for (int i=0; i<index_greaterthan; i++) {
        SSTF_order[index_lessthan+1+i] = SSTF_greaterthan[i];
    }
 

    int head_movements = 0;
    head_movements = (init_head_position-sorted_array[0])+(sorted_array[REQUESTS-1]-sorted_array[0]);


    for (int i=0; i<REQUESTS; i++) {
        if (i == REQUESTS - 1) {
            printf("%d \n", SSTF_order[i]);
            break;
        }
        printf("%d ", SSTF_order[i]);
    }
        
    return head_movements;

}



// SCAN function
int SCAN(int head_position, int head_index, int sorted_array[], int direction, int top, int bottom) {

    int SCAN_order[REQUESTS];
    int head_movements;
    int init_head_position = head_position;

    int head_in_requests;       // 0 is in, 1 is not in
    for (int i=0; i<REQUESTS; i++) {
        if (head_position == sorted_array[i]) {
            head_in_requests = 0;
            break;
        }
        head_in_requests = 1;
    }


    // left direction
    if (direction == 0) {

        // adjust head position if initial one is not in request array
        if (head_in_requests == 1) {
            int min = head_position;
            int min_index, diff;
            for(int i=0; i<REQUESTS; i++){
                diff = head_position - sorted_array[i];
                if (diff < min && diff > 0) {
                    min = diff;
                    min_index = i;
                }
            }
            head_position = sorted_array[min_index];
            head_index = min_index;
            SCAN_order[0] = sorted_array[min_index];

        }
        else {
            SCAN_order[0] = head_position;
        }

        // order for SCAN
        for (int i=1; i<=head_index; i++) {
            SCAN_order[i] = sorted_array[head_index-i];
        }
        for (int i=head_index+1; i<REQUESTS; i++) {
            SCAN_order[i] = sorted_array[i];
        }
        head_movements = (init_head_position-bottom)+(SCAN_order[REQUESTS-1]-bottom) ;

    }

    // right direction
    if (direction == 1) {

        // adjust head position if not in request array
        if (head_in_requests == 1) {
            int min = head_position;
            int min_index, diff;
            for(int i=0; i<REQUESTS; i++){
                diff = head_position - sorted_array[i];
                if (diff < min && diff > 0) {
                    min = diff;
                    min_index = i;
                }
            }
            head_position = sorted_array[min_index+1];
            head_index = min_index+1;
            SCAN_order[0] = sorted_array[min_index+1];
        }
        else {
            SCAN_order[0] = head_position;
        }


        for (int i=1; i<=REQUESTS-head_index; i++) {
            SCAN_order[i] = sorted_array[head_index+i];
        }
        for (int i=0; i<head_index; i++) {
            SCAN_order[REQUESTS-head_index+i] = sorted_array[head_index-1-i];
        }
        head_movements = (top-init_head_position)+(top-SCAN_order[REQUESTS-1]);
    }


    for (int i=0; i<REQUESTS; i++) {
        if (i == REQUESTS - 1) {
            printf("%d \n", SCAN_order[i]);
            break;
        }
        printf("%d ", SCAN_order[i]);
    }

    return head_movements;


}



// C-SCAN function
int C_SCAN(int head_position, int head_index, int sorted_array[], int direction, int top, int bottom) {

    int C_SCAN_order[REQUESTS];
    int head_movements;
    int init_head_position = head_position;

    // check if head is in request array
    int head_in_requests;       // 0 is in, 1 is not in
    for (int i=0; i<REQUESTS; i++) {
        if (head_position == sorted_array[i]) {
            head_in_requests = 0;
            break;
        }
        head_in_requests = 1;
    }


    // left direction
    if (direction == 0) {

        if (head_in_requests == 1) {
            int min = head_position;
            int min_index, diff;
            for(int i=0; i<REQUESTS; i++){
                diff = head_position - sorted_array[i];
                if (diff < min && diff > 0) {
                    min = diff;
                    min_index = i;
                }
            }
            head_position = sorted_array[min_index];
            head_index = min_index;
            C_SCAN_order[0] = sorted_array[min_index];

        }
        else {
            C_SCAN_order[0] = head_position;
        }


        for (int i=1; i<=head_index; i++) {
            C_SCAN_order[i] = sorted_array[head_index-i];
        }
        for (int i=1; i<REQUESTS-head_index; i++) {
            C_SCAN_order[head_index+i] = sorted_array[REQUESTS-i];
        }
        head_movements = (init_head_position-bottom)+(top-bottom)+(top-C_SCAN_order[REQUESTS-1]);

    }


    // right direction
    if (direction == 1) {

        if (head_in_requests == 1) {
            int min = head_position;
            int min_index, diff;
            for(int i=0; i<REQUESTS; i++){
                diff = head_position - sorted_array[i];
                if (diff < min && diff > 0) {
                    min = diff;
                    min_index = i;
                }
            }
            head_position = sorted_array[min_index+1];
            head_index = min_index+1;
            C_SCAN_order[0] = sorted_array[min_index+1];
        }
        else {
            C_SCAN_order[0] = head_position;
        }


        for (int i=1; i<=REQUESTS-head_index; i++) {
            C_SCAN_order[i] = sorted_array[head_index+i];
        }
        for (int i=0; i<head_index; i++) {
            C_SCAN_order[REQUESTS-head_index+i] = sorted_array[i];
        }
        head_movements = (top-init_head_position)+(top-bottom)+(C_SCAN_order[REQUESTS-1]-bottom);
    }


    for (int i=0; i<REQUESTS; i++) {
        if (i == REQUESTS - 1) {
            printf("%d \n", C_SCAN_order[i]);
            break;
        }
        printf("%d ", C_SCAN_order[i]);
    }

    return head_movements;

}




// LOOK FUNCTION
int LOOK(int head_position, int head_index, int sorted_array[], int direction) {

    int LOOK_order[REQUESTS];
    int head_movements;
    int init_head_position = head_position;

    int head_in_requests;       // 0 is in, 1 is not in
    for (int i=0; i<REQUESTS; i++) {
        if (head_position == sorted_array[i]) {
            head_in_requests = 0;
            break;
        }
        head_in_requests = 1;
    }


    // left direction
    if (direction == 0) {

        if (head_in_requests == 1) {
            int min = head_position;
            int min_index, diff;
            for(int i=0; i<REQUESTS; i++){
                diff = head_position - sorted_array[i];
                if (diff < min && diff > 0) {
                    min = diff;
                    min_index = i;
                }
            }
            head_position = sorted_array[min_index];
            head_index = min_index;
            LOOK_order[0] = sorted_array[min_index];

        }
        else {
            LOOK_order[0] = head_position;
        }


        for (int i=1; i<=head_index; i++) {
            LOOK_order[i] = sorted_array[head_index-i];
        }
        for (int i=head_index+1; i<REQUESTS; i++) {
            LOOK_order[i] = sorted_array[i];
        }
        head_movements = (init_head_position-sorted_array[0])+(sorted_array[REQUESTS-1]-sorted_array[0]);

    }

    // right direction
    if (direction == 1) {

        if (head_in_requests == 1) {
            int min = head_position;
            int min_index, diff;
            for(int i=0; i<REQUESTS; i++){
                diff = head_position - sorted_array[i];
                if (diff < min && diff > 0) {
                    min = diff;
                    min_index = i;
                }
            }
            head_position = sorted_array[min_index+1];
            head_index = min_index+1;
            LOOK_order[0] = sorted_array[min_index+1];
        }
        else {
            LOOK_order[0] = head_position;
        }


        for (int i=1; i<=REQUESTS-head_index; i++) {
            LOOK_order[i] = sorted_array[head_index+i];
        }
        for (int i=0; i<head_index; i++) {
            LOOK_order[REQUESTS-head_index+i] = sorted_array[head_index-1-i];
        }
        head_movements = (sorted_array[REQUESTS-1]-init_head_position) + (sorted_array[REQUESTS-1]-sorted_array[0]);
    }


    for (int i=0; i<REQUESTS; i++) {
        if (i == REQUESTS - 1) {
            printf("%d \n", LOOK_order[i]);
            break;
        }
        printf("%d ", LOOK_order[i]);
    }

    return head_movements;

}




// C-LOOK function
int C_LOOK(int head_position, int head_index, int sorted_array[], int direction) {

    int C_LOOK_order[REQUESTS];
    int head_movements;
    int init_head_position = head_position;

    int head_in_requests;       // 0 is in, 1 is not in
    for (int i=0; i<REQUESTS; i++) {
        if (head_position == sorted_array[i]) {
            head_in_requests = 0;
            break;
        }
        head_in_requests = 1;
    }

    // left direction
    if (direction == 0) {

        if (head_in_requests == 1) {
            int min = head_position;
            int min_index, diff;
            for(int i=0; i<REQUESTS; i++){
                diff = head_position - sorted_array[i];
                if (diff < min && diff > 0) {
                    min = diff;
                    min_index = i;
                }
            }
            head_position = sorted_array[min_index];
            head_index = min_index;
            C_LOOK_order[0] = sorted_array[min_index];

        }
        else {
            C_LOOK_order[0] = head_position;
        }


        for (int i=1; i<=head_index; i++) {
            C_LOOK_order[i] = sorted_array[head_index-i];
        }
        for (int i=1; i<REQUESTS-head_index; i++) {
            C_LOOK_order[head_index+i] = sorted_array[REQUESTS-i];
        }
        head_movements = (init_head_position-sorted_array[0])+(sorted_array[REQUESTS-1]-sorted_array[0])+(sorted_array[REQUESTS-1]-C_LOOK_order[REQUESTS-1]);
    }


    // right direction
    if (direction == 1) {

        if (head_in_requests == 1) {
            int min = head_position;
            int min_index, diff;
            for(int i=0; i<REQUESTS; i++){
                diff = head_position - sorted_array[i];
                if (diff < min && diff > 0) {
                    min = diff;
                    min_index = i;
                }
            }
            head_position = sorted_array[min_index+1];
            head_index = min_index+1;
            C_LOOK_order[0] = sorted_array[min_index+1];
        }
        else {
            C_LOOK_order[0] = head_position;
        }


        for (int i=1; i<=REQUESTS-head_index; i++) {
            C_LOOK_order[i] = sorted_array[head_index+i];
        }
        for (int i=0; i<head_index; i++) {
            C_LOOK_order[REQUESTS-head_index+i] = sorted_array[i];
        }
        head_movements = (sorted_array[REQUESTS-1]-init_head_position)+(sorted_array[REQUESTS-1]-sorted_array[0])+(C_LOOK_order[REQUESTS-1]-sorted_array[0]);
    }


    for (int i=0; i<REQUESTS; i++) {
        if (i == REQUESTS - 1) {
            printf("%d \n", C_LOOK_order[i]);
            break;
        }
        printf("%d ", C_LOOK_order[i]);
    }

    return head_movements;

}






int main(int argc, char *argv[]) {

    // opening request.bin and storing in array
    FILE* fptr = fopen("request.bin", "rb");


    if (fptr == NULL) {
        printf("Error opening file!");
        exit(1);
    }

    fread(requests_array, sizeof(int), REQUESTS, fptr);

    fclose(fptr);


    // checking command line inputs
    if (argc > 3 || argc < 3) {
        printf("Invalid number of arguments \n");
        return 0;
    }

    head_position = atoi(argv[1]);

    if (strcmp(argv[2], "LEFT") == 0) {
        direction = 0;
    }
    else if (strcmp(argv[2],"RIGHT") == 0) {
        direction = 1;
    }
    else {
        printf("Not a valid direction \n");
        return 0;
    }
    
    printf("Total Requests: %d \n", REQUESTS);
    printf("Initial Head Position: %d \n", head_position);
    printf("Direction of Head: %s \n", argv[2]);
    printf("\n");


    // requests array sorted in ascending order
    int sorted_array[REQUESTS];
    int head_index;
    int top = 299;
    int bottom = 0;

    for (int i=0; i<REQUESTS; i++) {
        sorted_array[i] = requests_array[i];
    }

    for (int i=0; i<REQUESTS; i++) {
        for (int j=i+1; j<REQUESTS; j++) {
            if (sorted_array[i] > sorted_array[j]) {
                int temp = sorted_array[i];
                sorted_array[i] = sorted_array[j];
                sorted_array[j] = temp;
            }
        }
    }

    // find the index of the head if it's in the request array
    for (int i=0; i<REQUESTS; i++) {
        if (sorted_array[i] == head_position) {
            head_index = i;
            break;
        }
    }


    // For FCFS
    printf("FCFS DISK SCHEDULING ALGORITHM \n");
    int FCFS_head_movements = FCFS(head_position, requests_array);
    printf("FCFS - Total Head Movements = %d \n", FCFS_head_movements);
    printf("\n");

    // For SSTF
    printf("SSTF DISK SCHEDULING ALGORITHM \n");
    int SSTF_head_movements = SSTF(head_position, requests_array, sorted_array);
    printf("SSTF - Total Head Movements = %d \n", SSTF_head_movements);
    printf("\n");

    // For SCAN
    printf("SCAN DISK SCHEDULING ALGORITHM \n");
    int SCAN_head_movements = SCAN(head_position, head_index, sorted_array, direction, top, bottom);
    printf("SCAN - Total Head Movements = %d \n", SCAN_head_movements);
    printf("\n");

    // For C-SCAN
    printf("C-SCAN DISK SCHEDULING ALGORITHM \n");
    int C_SCAN_head_movements = C_SCAN(head_position, head_index, sorted_array, direction, top, bottom);
    printf("C-SCAN - Total Head Movements = %d \n", C_SCAN_head_movements);
    printf("\n");

    // For LOOK
    printf("LOOK DISK SCHEDULING ALGORITHM \n");
    int LOOK_head_movements = LOOK(head_position, head_index, sorted_array, direction);
    printf("LOOK - Total Head Movements = %d \n", LOOK_head_movements);
    printf("\n");

    // FOR C-LOOK
    printf("C-LOOK DISK SCHEDULING ALGORITHM \n");
    int C_LOOK_head_movements = C_LOOK(head_position, head_index, sorted_array, direction);
    printf("C-LOOK - Total Head Movements = %d \n", C_LOOK_head_movements);
    printf("\n");

    return 0;



}






