/*
*
* Author: Alex Bartella (bartella, 400308868) & Jacqueline Leung (leungw18, 400314153)
* Lab test 2
* Group 29, L03
*/
#include <stdio.h>
#include <sys/mman.h> /*For mmap() function*/
#include <string.h> /*For memcpy function*/
#include <fcntl.h> /*For file descriptors*/
#include <stdlib.h> /*For file descriptors*/
#include <unistd.h>


#define OFFSET_MASK	127	// # of bits for offset mask = # of bits for page size - 1
#define PAGES 10		// 8 pages of memory
#define OFFSET_BITS 7 
#define PAGE_SIZE 128		//2^7 page size 

#define INT_SIZE 4
#define INT_COUNT 10
#define MEMORY_SIZE INT_COUNT*INT_SIZE



signed char *mmapfptr;
int page_table[10];

int intArray[MEMORY_SIZE];


int main(int argc, char *argv[]) {
	unsigned int page_number;
	unsigned int frame_number;
	int offset;
	unsigned int virtual_address;
	unsigned int physical_address;
	char buff[10];

	int mmapfile_fd = open("page_table.bin", O_RDONLY);
    
    mmapfptr = mmap(0, MEMORY_SIZE, PROT_READ, MAP_PRIVATE, mmapfile_fd, 0);

    for (int i=0; i<INT_COUNT; i++) {
        memcpy(page_table + i, mmapfptr + 4*i, INT_SIZE);
    }

	FILE *fptr = fopen("ltaddr.txt", "r");


	while (fgets(buff,10,fptr) != NULL) {
		
		virtual_address = atoi(buff);

		page_number = virtual_address >> OFFSET_BITS;

		offset = virtual_address & OFFSET_MASK;

		frame_number = page_table[page_number];

		physical_address = (frame_number << OFFSET_BITS) | offset;

		printf("Virtual addr is %d: Page# = %d & Offset = %d frame number = %d Physical addr is %d. \n", virtual_address, page_number, offset, frame_number, physical_address);


	}

	fclose(fptr);
	munmap(mmapfptr, MEMORY_SIZE);


	return 0;
}
