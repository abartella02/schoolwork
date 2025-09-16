-- Alex Bartella 400308868
-- Assignment 1

CREATE TABLE Person(
    personID    int PRIMARY KEY NOT NULL,
    name        char(20),
    age         int,
    phoneNum    varchar(12)
);

CREATE TABLE Passenger(
    personID       int NOT NULL REFERENCES Person(personID),
    dietPref       varchar(15)
);

CREATE TABLE Pilot(
    personID    int PRIMARY KEY NOT NULL REFERENCES Person(personID),
    position    varchar(15),
    salary      int
);

CREATE TABLE CabinCrew(
    personID       int PRIMARY KEY NOT NULL REFERENCES Person(personID),
    position       varchar(15),
    salary         int
);

CREATE TABLE GroundStaff(
    personID    int PRIMARY KEY NOT NULL REFERENCES Person(personID),
    department  varchar(25),
    salary      int
);

CREATE TABLE Ticket(
    ticketNum   int PRIMARY KEY NOT NULL,
    seatNum     int,
    class       varchar(1)
);

CREATE TABLE Airline(
    airlineName     varchar(20),
    alias           varchar(2) PRIMARY KEY NOT NULL
);

CREATE TABLE Airplane(
    serialNum   int PRIMARY KEY NOT NULL,
    manuf       varchar(20),
    model       varchar(20)
);

CREATE TABLE Country(
    code        varchar(3) PRIMARY KEY NOT NULL,
    name        varchar(20),
    continent   varchar(20)
);

CREATE TABLE Airport(
    IATAcode    varchar(3) PRIMARY KEY NOT NULL,
    name        varchar(20),
    city        varchar(20)
);

CREATE TABLE Route(
    routeID     int PRIMARY KEY NOT NULL,
    numStops    int,
    duration    time
);

CREATE TABLE ScheduledFlights(
    flightNum           varchar(20) PRIMARY KEY NOT NULL,
    departDate          date,
    arriveDate          date,
    schedDepartTime     time,
    schedArriveTime     time,
    actualDepartTime    time,
    actualArriveTime    time
);

CREATE TABLE Baggage(
    type        varchar(20) NOT NULL,
    quantity    int,
    weight      real,
    fragile     int,  -- int 1 or 0 for true or false
    ticketNum   int NOT NULL,
    FOREIGN KEY (ticketNum) REFERENCES Ticket(ticketNum),
    PRIMARY KEY (ticketNum, type)
);


--------------------
-- RELATIONS
--------------------

-- Passenger buys Ticket
CREATE TABLE Buys(
    personID    int NOT NULL REFERENCES Person(personID),
    ticketNum   int PRIMARY KEY NOT NULL REFERENCES Ticket(ticketNum),
    price       int
);

-- Pilot flies Airplane
CREATE TABLE Flies(
    personID    int NOT NULL REFERENCES Pilot(personID),
    planeSNum   int NOT NULL REFERENCES Airplane(serialNum),
    PRIMARY KEY (personID, planeSNum)
);

-- GroundStaff works for Airport
CREATE TABLE GroundWorksFor(
    IATAcode    varchar(3) REFERENCES Airport(IATAcode),
    staffID     int PRIMARY KEY NOT NULL REFERENCES GroundStaff(personID)
);

-- CabinCrew works for Airline
CREATE TABLE CabinWorksFor(
    airlineAlias    varchar(2) REFERENCES Airline(alias),
    crewID          int PRIMARY KEY NOT NULL REFERENCES CabinCrew(personID)
);

-- Airline owns Airplanes
CREATE TABLE Owns(
    airlineAlias    varchar(2) REFERENCES Airline(alias),
    planeSNum       int PRIMARY KEY NOT NULL REFERENCES Airplane(serialNum)
);

-- Airport in country
CREATE TABLE In(
    IATAcode    varchar(3) PRIMARY KEY NOT NULL REFERENCES Airport(IATAcode),
    countryCode varchar(3) NOT NULL REFERENCES Country(code)
);

-- Airline belongs to a country
CREATE TABLE BelongsTo(
    airlineAlias    varchar(2) PRIMARY KEY NOT NULL REFERENCES Airline(alias),
    countryCode     varchar(3) NOT NULL REFERENCES Country(code)
);

-- Airline is the source of a Route
CREATE TABLE Source(
    routeID     int PRIMARY KEY NOT NULL REFERENCES Route(routeID),
    IATAcode    varchar(3)
);

-- Airline is the destination of a route
CREATE TABLE Dest(
    routeID     int PRIMARY KEY NOT NULL REFERENCES Route(routeID),
    IATAcode    varchar(3)
);

-- Airline has route
CREATE TABLE Has(
    routeID         int REFERENCES Route(routeID),
    airlineAlias    varchar(2) REFERENCES Airline(alias)
);

-- Scheduled flight contains route
CREATE TABLE Contains(
    flightNum   varchar(20) PRIMARY KEY NOT NULL REFERENCES ScheduledFlights(flightNum),
    routeID     int REFERENCES Route(routeID)
);

-- Airline uses scheduled flight
CREATE TABLE Uses(
    airlineAlias    varchar(2) NOT NULL REFERENCES Airline(alias),
    flightNum       varchar(20) PRIMARY KEY NOT NULL REFERENCES ScheduledFlights(flightNum)
);

-- Ticket bought for scheduled flight
CREATE TABLE BoughtFor(
    flightNum       varchar(20) NOT NULL REFERENCES ScheduledFlights(flightNum),
    ticketNum       int PRIMARY KEY NOT NULL REFERENCES Ticket(ticketNum)
);
