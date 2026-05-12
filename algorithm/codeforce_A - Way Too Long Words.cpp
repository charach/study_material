#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(void)
{
    int n;
    scanf("%d",&n);
    char *str = (char*)malloc(105*sizeof(char));
    for( unsigned int i = 0 ; i < n ; i ++){
        memset(str, 0 , 105*sizeof(char));
        scanf("%s",str);
        if ( strlen(str) < 10){
            printf("%s\n",str);
        }else{
            printf("%c%lu%c\n",str[0],strlen(str)-2, str[strlen(str)-1]);
        }
    }
    free(str);
    
}