#include <cmath>
#include <iostream>
using namespace std;

/** contains functions that return structs and structs as arguments
 * calculating distance between 2-variable coordinate in Euclidean space
 * @author Dave Amiana
 */

typedef struct point {
  point(double x_arg, double y_arg) {
    x = x_arg;
    y = y_arg;
  }
  double x = 0.0;
  double y = 0.0;
} point;

/** formula for calculating the distance:
 * sqrt((pow,abs(x1-x2),2)+(pow,abs(y1-y2),2))
 */
double distance(point a, point b) {
  double x_dist = abs(b.x - a.x);
  double y_dist = abs(b.y - a.y);
  return sqrt(pow(x_dist, 2) + pow(y_dist, 2));
}

// resolve function that ought to return a struct
point getMiddle(point a, point b) {
  point m;
  m.x = (a.b + b.x) / 2;
  m.y = (a.y + b.y) / 2;
  return m;
}

int main() {
  point p1{1, 2};
  point p2{3, 4};
  point middle;
  middle = getMiddle(p1, p2);
  std::cout << "distance: " << distance(p1, p2);
  std::cout << "middle point: " << middle;
  return 0;
}