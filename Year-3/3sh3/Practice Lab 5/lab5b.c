/**
* Author: Prof. Neerja Mhaskar
* Course: Operating Systems CS 3SH3/SFWRENG
*/

#include <stdio.h>
#include <sys/mman.h> /*For mmap() function*/
#include <string.h> /*For memcpy function*/
#include <fcntl.h> /*For file descriptors*/
#include <stdlib.h> /*For file descriptors*/

/*Define the needed macros*/

// Define character pointer to store the starting address of the memory mapped file

//Define integer array to store numbers from the memory mapped file.


int main(int argc, const char *argv[])
{   
    //Use the open() system call to open and read the numbers.bin file.
    
    //Use the mmap() system call to memory map numbers.bin file
   

    //Use a for loop to retrieve the contents of the memory mapped file and store it in the integer array using memcpy() function.
    
    //Sum up all the numbers
    //Print Sum
    //unmap the memory file
    return 0;
}
