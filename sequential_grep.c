#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FILE_LINE 5120000
#define LINE_WIDTH 128

//self defined strstr function, to locate the searching word
char *my_strstr(const char *str1, const char *str2)
{
    char *cp = (char *)str1;
    char *s1, *s2;
	
	//if searching word is empty, return the sentence
    if(!*str2)
        return ((char*)str1);

    int i = 0;
	//keep searching until reach the end of sentence
    while(i < LINE_WIDTH)
    {
        s1 = cp;
        s2 = (char *)str2;

        while(*s1 && *s2 && !(*s1 - *s2))
            s1++, s2++;

        if(!*s2)
            return cp;
        cp++;
        i++;
    }

    return NULL;
}

int main(int argc, char *argv[])
{
	//get arguments from command line
	char *Filename = argv[1];
	char *Regexp = argv[2];
    if(Regexp==NULL||Filename==NULL){
        printf("Usage: ./program [file name] [searching words]");
        return -1;
    }
	//open file
    FILE *f;
    f = fopen(Filename, "r");
    if(f == NULL)
    {
        printf("Fail to open file!\n");
        return -1;
    }
	
	//store file content
    char **file;
    int i;

    file = (char **)malloc(sizeof(char*)*FILE_LINE);
    //keep the continuity of memory
    file[0] = (char *)malloc(sizeof(char)*FILE_LINE*LINE_WIDTH);
    for(i = 1; i < FILE_LINE; i++)
        file[i] = file[i-1] + LINE_WIDTH;

	//read file
    for(i = 0; i < FILE_LINE; i++)
    {
        fgets(file[i], LINE_WIDTH, f);
    }

    int offset = 0;
    char *pch;
    char *result = (char *)malloc(sizeof(char)*FILE_LINE*LINE_WIDTH);
	//check whether there is searching word in each line
    for(i = 0; i < FILE_LINE; i++)
    {
        pch = my_strstr(file[i], Regexp);
        if(pch != NULL)
        {
            memcpy(&result[offset*LINE_WIDTH], file[i], sizeof(char)*LINE_WIDTH);
            offset++;
        }
    }
	
	//print result
    for(i = 0; i < FILE_LINE; i++)
    {
        if(&result[i*LINE_WIDTH]!=NULL)
            printf("%s", &result[i*LINE_WIDTH]);
    }
	
	//free memory
	free(file);
	free(result);
	
    return 0;
}
