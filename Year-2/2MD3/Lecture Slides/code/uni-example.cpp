#include <iostream>
#include <string>
using namespace std;

class Person {  // Person (base class) 
    private:
        string name;   // name
        string idNum;  // university ID number

    public:
        Person(const string& nm, const string& id);
        virtual ~Person();
        virtual void print();      // print information
        string getName();          // retrieve name
};

Person::Person(const string& nm, const string& id)
    : name(nm), idNum(id)  // initialize name and id
    { }

Person::~Person(){
    cout << "~Person() " << name << endl;
}

void Person::print() { // definition of Person print
    cout << "Name " << name << ", " << "IDnum " << idNum << endl;
}

string Person::getName() { // definition of Person getName
    return name;
}


class Student : public Person {  // Student (derived from Person)
    private:
        string major;  // major subject
        int gradYear;  // graduation year
        
    public:
        Student(const string& nm, const string& id, const string& maj, int year);
        ~Student();
        void print();                              // print information
        void changeMajor(const string& newMajor);  // change major  
};


Student::Student(const string& nm, const string& id, const string& maj, int year)
    : Person(nm, id), major(maj), gradYear(year)
    { }

Student::~Student(){
    cout << "~Student()" << endl;
}

void Student::print() {   // definition of Student print 
    Person::print();      // first print Person information
    cout << "Major " << major << ", Year " << gradYear << endl;  // then student-specific info
}

void Student::changeMajor(const string& newMajor) {  // definition of Student print 
    major = newMajor;
}

class Professor : public Person {  // Professor (derived from Person)
    private:
        string department;  // department name
        string office;  // office number
        
    public:
        Professor(const string& nm, const string& id, const string& dep, const string& offc);
        ~Professor();
        void print();                              // print information
};

Professor::Professor(const string& nm, const string& id, const string& dep, const string& offc)
    : Person(nm, id), department(dep), office(offc)
    { }

Professor::~Professor(){
    cout << "~Professor()" << endl;
}

void Professor::print() {   // definition of Professor's print 
    Person::print();      // first print Person information
    cout << "Department " << department << ", Office " << office << endl;  // then Professor-specific info
}

int main() {
    Person person("Mary", "12-345");  // declare a Person
    //Student student("Bob", "98-764", "Math", 2020);  // declare a Student
    //Professor prof("John", "22-224", "CAS", "ITB223");  // declare a Professor
    Person *person_ptr = new Student("Alice", "34-875", "CS", 2021);  // declare a Student dynamically

    person.print();   // invokes Person::print()

    person_ptr -> print();  // invokes Student::print()

    cout << "\nList of people in the Univeristy with insurance problem:" << endl;

    Person *people_with_insurance_problem[4];
    people_with_insurance_problem[0] = new Student ("Bob", "98-764", "Math", 2020);  // declare a Student
    people_with_insurance_problem[1] = new Professor ("John", "22-224", "CAS", "ITB223");  // declare a Professor
    people_with_insurance_problem[2] = &person;
    people_with_insurance_problem[3] = person_ptr;

    for (int i = 0; i < 4; i++) {
        cout << "\nPerson " << i << endl;
        people_with_insurance_problem[i]->print();
    }

    cout << "\n\nDeleting some people :" << endl;
    delete people_with_insurance_problem[0];

    return EXIT_SUCCESS;
}
