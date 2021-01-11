#pragma once
# include<iostream>

// namespaces exist to avoid naming conflicts
// to access namespace use the syntax namespace::<nameFunction>
// or use the <use namespace <namespace>>
// namespace are like classes of their own 
namespace Name{
    /* - adds intger a and b. */
    int add(int a, int b){return a+b;}
    /* - subtracts integer a and b**/
    int dif(int a, int b){return a-b;}
    /* - divides integer a and b**/
    float div(float a, float b){return a/b;}
    /* - multiplies integer a and b**/
    float mult(float a, float b){return a*b;}
}