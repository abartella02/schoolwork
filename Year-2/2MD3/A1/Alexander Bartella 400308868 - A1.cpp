//******************************************//
//Alexander Bartella, Student ID: 400308868
//******************************************//

#include <iostream>
#include <math.h>
using namespace std;

//declarations-----------------------------------------------------------
class Point {
    public:
        Point();
        Point(double temp_x, double temp_y);
        double getX();
        double getY();
        double distanceTo(Point otherP);

        friend istream& operator>>(istream& i, Point& pt);
        
        friend ostream& operator<<(ostream& o, Point& pt);

    private:
        double x;
        double y;
};

class Polygon {
    public:
        Polygon(int n=10);
        Polygon(int n, Point* point_list);
        ~Polygon(); //destructor
        double perimeter();
        friend istream& operator>>(istream& i, Polygon& py);
        
        friend ostream& operator<<(ostream& o, Polygon& py);

        int getnumpoints(); //remove later
    
    private:
        int numPoints;
        Point* points;
};

//definitions------------------------------------------------------

//Point-------------------------
Point::Point(){ //default constructor
    x = 0; //initialize x and y to zero
    y = 0;
}

Point::Point(double temp_x, double temp_y){ //2-argument constructor
    x = temp_x; //set x and y to constructor arguments
    y = temp_y;
}

double Point::getX(){ //get value of x
    return x;
}

double Point::getY(){ //get value of y
    return y;
}

double Point::distanceTo(Point otherP){ //distance to other point
    double dist = 0; //default distance
    double deltax = x - otherP.x; //difference between x values
    double deltay = y - otherP.y; //difference between y valyes
    dist = sqrt(pow(deltax, 2) + pow(deltay, 2)); //substitute delta of x and y into distance formula

    return dist;
}

istream& operator>>(istream& i, Point& pt){ //overloading >> operator
    cout << "Enter x: "; //prompt
    i >> pt.x; //assign input to x
    cout << "Enter y: "; //repeat for y
    i >> pt.y;

    return i;
}

ostream& operator<<(ostream& o, Point& pt){ //overloading << operator
    o << "P(" << pt.x << ", " << pt.y << ")"; //printing point in correct format

    return o;
}

//Polygon------------------------------
Polygon::Polygon(int n){ //single argument constructor
    numPoints = n; //assign value of numPoints (default is 10)
    points = new Point[numPoints](); //declare dynamically allocated list
}

Polygon::Polygon(int n, Point* point_list){ //2-argument constructor
    numPoints = n;
    points = point_list; 
}

Polygon::~Polygon(){ //destructor
    delete[] points; //delete dynamically allocated list
}

double Polygon::perimeter(){ //perimeter
    double length = 0; //default perimeter is 0
    for(int i=0;i<numPoints-1;i++){ //iterate through points
        length += points[i].distanceTo(points[i+1]); //find distance to next point and add to total
    }
    length+=points[numPoints-1].distanceTo(points[0]); //add distance from last to first point

    return length;
}

istream& operator>>(istream& i, Polygon& py){ //overload >>
    char confirm;
    int counter = 0;
    double X = 0, Y = 0; //default values for X and Y (if nothing is inputted)
    do{
        if((counter+1) > py.numPoints){ //only allows for only the approprtate amount of points to be entered
            cout << "Maximum number of points entered." << endl;
            break;
        }

        cout << "Enter point's x value: "; //prompt
        i >> X; //assign input to X
        cout << "Enter point's y value: "; //repeat for Y
        i >> Y;

        py.points[counter] = Point(X, Y); //create point using inputs and append to list of points

        cout << "Enter another point? (y/n): "; //prompt to enter another point
        cin >> confirm;

        counter++;

    }while(confirm == 'y'); //breaks loop if user does not enter 'y' after entering a point
    
    return i;
}

ostream& operator<<(ostream& o, Polygon& py){ //overload <<
    for(int i = 0; i<py.numPoints; i++){ //iterating through list of points
        o << py.points[i] << " "; //print each point in the list
    }

    return o;
}

//main---------------------------------------------------------------------------------
int main() {
    //test code
    Polygon poly(4);
    cin >> poly; //enter P(1, 1) P(1, 2) P(2, 2) P(2, 1)
    double perimeter = poly.perimeter();
    cout << poly << "With Perimeter: " << perimeter <<endl;
    //above should print P(1, 1) P(1, 2) P(2, 2) P(2, 1) With Perimeter: 4
    
    return 0;
}