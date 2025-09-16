-- Alex Bartella 400308868
-- SFWRENG 3DB3 Assignment 2

-- Q1
SELECT per.personID, per.Name, per.Age
FROM Passenger AS pa, Person AS per
WHERE pa.PersonID = per.PersonID 
    AND per.Age >= 20 AND per.Age <= 30 
    AND (pa.DietaryPref = 'Vegan' OR pa.DietaryPref = 'Vegetarian');

-- Q2a
SELECT Model, COUNT(Model) AS cnt
FROM Airplane
GROUP BY Model;

-- Q2b
SELECT a.Name, p.Model, COUNT(p.Model) AS cnt
FROM Airplane AS p, Airline AS a
WHERE a.Alias = p.AirlineAlias 
    AND (a.Name = 'Air Canada' 
    OR a.Name = 'Etihad Airways' 
    OR a.Name = 'United Airlines')
GROUP BY a.Name, p.Model;

-- Q3a
SELECT t.TicketNo, AVG(b.TotalWeight) as avg_baggage_weight
FROM Ticket AS t, ScheduledFlight AS sf, Route AS r, Airline AS a, Use AS u, Baggage as b
WHERE 
    t.FlightNo = sf.FlightNo 
    AND t.FlightDepDate = sf.DepDate
    AND sf.RouteID = r.RouteID
    AND u.RouteID = r.RouteID 
    AND u.AirlineAlias = a.Alias
    AND u.AirlineAlias = sf.AirlineAlias
    AND a.Name = 'Air Canada'
    AND b.TicketNo = t.TicketNo
GROUP BY t.TicketNo;


-- Q3b
SELECT b.TicketNo, SUM(b.TotalWeight) AS Total_Weight
FROM Ticket AS t, Baggage AS b, 
    (
        SELECT FlightNo, DepDate
        FROM ScheduledFlight
        WHERE ArrDate >= '20231210' AND DepDate <= '20240103'
    ) as sf
WHERE 
    b.TicketNo = t.TicketNo 
    AND NOT b.Fragile 
    AND b.BagType = 'Oversized' 
    AND sf.FlightNo = t.FlightNo
    AND sf.DepDate = t.FlightDepDate
GROUP BY b.TicketNo
HAVING SUM(b.TotalWeight) > 90;

-- Q4
SELECT b.TicketNo, t.FlightDepDate, b.Price, b.Website
FROM Book as b, Ticket t, ScheduledFlight as sf, Route as r
WHERE b.ticketNo = t.TicketNo AND t.FlightNo = sf.FlightNo AND t.FlightDepDate = sf.DepDate 
AND sf.RouteID = r.RouteID and r.srcAirport = 'YYZ' AND r.dstAirport = 'MCO'
AND b.Price = (
    SELECT MIN(b1.Price)
    FROM Ticket AS t1, ScheduledFlight as sf1, Route as r1, Book as b1
    WHERE t1.FlightNo = sf1.FlightNo AND t1.FlightDepDate = sf1.DepDate
    AND sf1.RouteID = r1.RouteID AND r1.srcAirport = 'YYZ' AND r1.dstAirport = 'MCO'
    AND b1.TicketNo = t1.TicketNo
);

-- Q5a
SELECT RouteID, COUNT(airlineAlias) AS airlines
FROM Use AS u
GROUP BY RouteID
HAVING COUNT(airlineAlias) >=3
ORDER BY COUNT(airlineAlias) desc;

-- Q5b
SELECT RouteID, srcAirport, dstAirport
FROM Route
WHERE NOT RouteID IN (
    SELECT u.RouteID
    FROM Use as u
);

-- Q6a
SELECT COUNT(DISTINCT pass.PersonID) AS NumStaffPassengers
FROM Passenger AS pass
JOIN (
    SELECT PersonID FROM Pilot
    UNION
    SELECT PersonID FROM CabinCrew
    UNION
    SELECT PersonID FROM GroundStaff
) AS p ON pass.PersonID = p.PersonID;

-- Q6b
SELECT Alias, COUNT(DISTINCT x.PersonID) as cnt
FROM Airline AS al
JOIN (
    (
        SELECT cc.PersonID, cc.AirlineAlias
        FROM CabinCrew as cc 
    )
    UNION
    (
        SELECT f.PilotID as PersonID, ap.AirlineAlias
        FROM Airplane AS ap
        JOIN Flies AS f ON f.AirplaneSNo = ap.SerialNo
    )
) AS x ON al.Alias = x.AirlineAlias
JOIN Passenger as p ON x.PersonID = p.PersonID
GROUP BY Alias;

-- Q7a
SELECT r.routeID, r.srcAirport, r.dstAirport
FROM Airline AS a
JOIN Use u ON a.Alias = u.AirlineAlias
JOIN Route r ON u.RouteID = r.RouteID
WHERE a.alias = 'ACA' AND NOT EXISTS (
    SELECT *
    FROM Airline AS a1
    JOIN Use u1 ON a1.Alias = u1.AirlineAlias
    JOIN Route r1 ON u1.RouteID = r1.RouteID
    WHERE 
        r.srcAirport = r1.dstAirport 
        AND r.dstAirport = r1.srcAirport
        AND a1.alias = 'ACA'
);

-- Q7b
SELECT x.routeID, x.srcAirport, x.dstAirport, x.TicketsSold
FROM 
(
    SELECT r.RouteID, r.srcAirport, r.dstAirport, COUNT(t.ticketNo) AS TicketsSold
    FROM Route AS r
    JOIN ScheduledFlight sf ON sf.RouteID = r.RouteID
    JOIN Ticket t ON t.FlightNo = sf.FlightNo AND t.FlightDepDate = sf.DepDate
    WHERE t.FlightDepDate >= '20231201' AND t.FlightDepDate <= '20231231'
    GROUP BY r.RouteID, r.srcAirport, r.dstAirport
) as x
WHERE x.TicketsSold = (
    SELECT MAX(TicketsSold)
    FROM (
        SELECT COUNT(t.ticketNo) AS TicketsSold
        FROM Route AS r
        JOIN ScheduledFlight sf ON sf.RouteID = r.RouteID
        JOIN Ticket t ON t.FlightNo = sf.FlightNo AND t.FlightDepDate = sf.DepDate
        WHERE t.FlightDepDate >= '20231201' AND t.FlightDepDate <= '20231231'
        GROUP BY r.RouteID
    )
);

-- Q8a
SELECT sf.FlightNo
FROM Ticket as t
JOIN ScheduledFlight sf ON sf.FlightNo = t.FlightNo AND sf.DepDate = t.FlightDepDate
JOIN Route r ON sf.RouteID = r.RouteID
WHERE sf.airlineAlias = 'ACA' AND r.srcAirport = 'YYZ' AND r.dstAirport = 'MCO' AND t.Class = 'First';

-- Q8b
SELECT a1.Alias, a1.Name, uc.Name
FROM Airline as a1
JOIN (
    SELECT c.Code, c.Name
    FROM Airline as a
    JOIN Country c ON a.CountryCode = c.Code
    GROUP BY c.Code, c.Name
    HAVING COUNT(a.Alias) = 1
) uc ON uc.code = a1.CountryCode;
