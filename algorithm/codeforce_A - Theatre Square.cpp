#include <stdio.h>
 
int main(void)
{
    unsigned long n,m,a;
    scanf("%lu %lu %lu",&n,&m,&a);
    printf("%lu\n", (n/a + (n%a == 0 ? 0:1)) *(m/a + (m%a == 0 ? 0:1)) );
}