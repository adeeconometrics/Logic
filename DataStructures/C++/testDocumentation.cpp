/**
 * CS0015L
 * @file testDocumentation.cpp
 * @author Dave Arthur D. Amiana
 * @version 0.1 01.07.2021
 */
#include "static.h"
#include <iostream>

using namespace std;

/**
 * Returns the sum of two integers
 * @param a addend
 * @param b addend
 * @return sum of a and b
 */
int sum(int a, int b) { return a + b; }

int main() {
  int x;
  int y;
  cin >> x >> y;
  cout << sum(x, y);
  if (x < y) {
    cout << "x is less than y";
  }
  cout << "no";
  return 0;
}