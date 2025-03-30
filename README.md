# EPEX day-ahead price prediction

This is a simple statistical model to predict EPEX day-ahead prices based on various parameters.
It works to a reasonably good degree. Better than many of the commercial solutions.
This repository includes
- The self-training prediction model itself
- A simple Flask app to get a REST API up
- A Dockerfile to easily have it running wherever with gunicorn

Supported Countries:
- Germany (default)
- Austria


## Lookout
- Maybe package it directly as a Home Assistant Add-on

## The Model
We sample a few sample points all over the country and fetch hourly [Weather data from Open-Meteo.com](https://open-meteo.com/) for those for the past n days (default n=90).
This serves as the main data source.



Parameters:

- Wind for each location
- Temperature for each location
- Expected solar output for an arbitrary fake solar installation (computed via OpenMeteoSolarForecast)
- Hour of day
- Whether it is a Holiday/Sunday
- Whether it is a Saturday

Output:
- Electricity price

## How it works
- First, we create a simple multi-linear-regression to get an idea how important each of
the training parameters is.
- This alone is not enough, because electricity prices are not linear.
E.g. low wind&solar leads to gas power plants being turned on, and due to merit order pricing, electricity prices explode.
- Therefore, we then multiply each parameter with its weight (LinReg factor) to get a "normalized" data set.
- In the next step, we use a KNN (k=3) approach to find hours in the past with similar properties and use that to determine the final price.

## Model performance
No scientific evaluation. I just looked at the result and they mostly seem to be within 1-10%.
Some observations:
- At night, predictions are usually within 1-2ct/kWh
- Morning/Evening peaks are usually within 3-4ct/kWh
- Extreme peaks due to "Dunkelflaute" are correctly detected, but estimation of the exact price is a challange. E.g.
the model might predict 75ct, while in reality it's only 60ct or vice versa
- High PV noons are usually correctly detected. Sometimes it will return 3ct instead of -1ct, but the ballpark is usually correct.

This graph compares the actual prices to the ones returned by the model for a random two week time period in early 2025.

Note that this was created for a time range in the past with historic weather data, rather than forecasted weather data,
so actual performance might be a bit worse if the weather forecast is not correct.

```mermaid
---
config:
    xyChart:
        width: 1700
        height: 900
        plotReservedSpacePercent: 80
        xAxis:
            showLabel: false
---
xychart-beta
    title "Performance comparison (Germany)"
    x-axis [1500,1501,1502,1503,1504,1505,1506,1507,1508,1509,1510,1511,1512,1513,1514,1515,1516,1517,1518,1519,1520,1521,1522,1523,1524,1525,1526,1527,1528,1529,1530,1531,1532,1533,1534,1535,1536,1537,1538,1539,1540,1541,1542,1543,1544,1545,1546,1547,1548,1549,1550,1551,1552,1553,1554,1555,1556,1557,1558,1559,1560,1561,1562,1563,1564,1565,1566,1567,1568,1569,1570,1571,1572,1573,1574,1575,1576,1577,1578,1579,1580,1581,1582,1583,1584,1585,1586,1587,1588,1589,1590,1591,1592,1593,1594,1595,1596,1597,1598,1599,1600,1601,1602,1603,1604,1605,1606,1607,1608,1609,1610,1611,1612,1613,1614,1615,1616,1617,1618,1619,1620,1621,1622,1623,1624,1625,1626,1627,1628,1629,1630,1631,1632,1633,1634,1635,1636,1637,1638,1639,1640,1641,1642,1643,1644,1645,1646,1647,1648,1649,1650,1651,1652,1653,1654,1655,1656,1657,1658,1659,1660,1661,1662,1663,1664,1665,1666,1667,1668,1669,1670,1671,1672,1673,1674,1675,1676,1677,1678,1679,1680,1681,1682,1683,1684,1685,1686,1687,1688,1689,1690,1691,1692,1693,1694,1695,1696,1697,1698,1699,1700,1701,1702,1703,1704,1705,1706,1707,1708,1709,1710,1711,1712,1713,1714,1715,1716,1717,1718,1719,1720,1721,1722,1723,1724,1725,1726,1727,1728,1729,1730,1731,1732,1733,1734,1735,1736,1737,1738,1739,1740,1741,1742,1743,1744,1745,1746,1747,1748,1749,1750,1751,1752,1753,1754,1755,1756,1757,1758,1759,1760,1761,1762,1763,1764,1765,1766,1767,1768,1769,1770,1771,1772,1773,1774,1775,1776,1777,1778,1779,1780,1781,1782,1783,1784,1785,1786,1787,1788,1789,1790,1791,1792,1793,1794,1795,1796,1797,1798,1799,1800,1801,1802,1803,1804,1805,1806,1807,1808,1809,1810,1811,1812,1813,1814,1815,1816,1817,1818,1819,1820,1821,1822,1823,1824,1825,1826,1827,1828,1829,1830,1831,1832,1833,1834,1835]
    line [7.1,11.7,13.1,14.0,12.5,10.8,10.1,8.0,7.8,7.9,7.9,8.1,8.6,10.3,13.5,15.3,13.5,10.1,6.8,5.1,3.1,2.8,5.0,7.0,9.9,14.3,17.7,17.0,14.2,11.9,10.8,9.3,9.2,9.0,9.2,8.9,9.5,11.1,13.9,13.7,10.7,7.2,2.5,0.1,-0.1,-0.0,0.6,5.0,9.6,14.1,17.1,16.7,14.2,12.3,11.6,10.3,10.7,10.2,10.1,10.2,10.7,12.3,15.8,14.8,10.8,8.0,2.7,-0.0,-0.1,-0.0,0.1,6.7,11.0,14.8,24.8,22.0,17.7,13.8,12.4,11.1,11.7,10.3,10.0,10.0,10.3,12.0,14.5,14.1,10.0,7.9,3.8,0.3,0.0,0.1,1.7,7.9,12.1,15.5,28.0,25.4,16.4,13.7,12.5,11.8,12.5,11.6,11.3,10.6,10.9,12.7,15.6,14.8,11.1,7.3,4.3,0.7,0.1,0.0,0.2,1.2,4.6,10.0,12.1,8.3,7.0,4.7,2.7,1.8,2.7,1.3,0.9,0.7,1.0,1.3,1.3,1.3,0.3,0.0,-0.2,-0.6,-1.5,-1.5,-0.7,-0.2,0.1,3.5,6.0,5.1,4.2,3.6,3.7,3.6,1.4,1.9,2.4,2.3,2.5,2.5,2.5,2.4,1.9,2.3,1.4,1.0,1.3,0.5,1.1,4.2,8.4,12.4,14.0,14.3,14.1,13.0,12.6,11.4,11.0,10.7,10.2,9.9,10.2,12.0,14.2,16.4,16.2,13.5,11.1,10.7,10.1,10.1,10.7,11.6,13.1,15.4,21.2,19.3,16.2,13.9,12.6,11.5,10.2,10.2,10.0,10.3,10.2,10.7,13.6,14.2,13.3,10.6,9.2,7.9,6.8,6.9,8.2,9.6,10.9,13.9,16.1,16.6,14.9,12.2,11.5,10.2,10.1,9.8,9.7,9.7,9.6,9.9,12.9,14.2,13.7,11.2,10.0,9.6,9.5,8.4,9.1,9.4,10.1,13.5,16.2,17.0,15.3,12.9,12.0,10.9,10.5,9.9,9.8,9.9,9.7,10.5,13.0,14.0,12.3,9.0,7.8,4.5,2.5,0.7,3.0,7.5,9.4,12.7,15.8,17.1,14.2,12.4,11.3,10.1,9.0,7.9,7.9,7.9,8.0,10.4,12.9,12.2,10.2,7.6,2.6,0.2,-0.0,-0.0,1.4,7.0,10.4,13.2,16.6,16.6,14.8,12.5,11.8,11.0,11.4,10.4,9.7,9.6,9.7,9.3,9.7,9.8,9.2,7.6,6.6,6.5,6.5,0.2,0.4,2.5,6.6,10.0,14.7,15.2,12.1,10.0,8.3,6.8,4.6,1.6,0.5,0.1,0.0,0.0,0.0,-0.0,-0.0,-0.3,-1.1,-1.9,-2.6,-2.6,-1.3,-0.4]
    line [7.9,13.1,14.3,13.4,12.2,12.4,12.2,10.7,8.9,9.4,9.0,9.3,9.1,11.3,14.0,15.9,14.4,9.6,4.0,5.6,7.2,6.5,2.9,7.7,10.9,15.2,17.3,16.7,14.8,12.4,12.0,10.6,10.6,10.6,10.5,10.2,10.3,12.3,13.7,17.5,10.8,8.4,4.0,1.7,-0.1,0.0,2.9,6.5,9.6,14.8,17.0,16.9,14.4,12.0,11.7,10.6,11.4,11.2,10.2,10.3,11.1,12.8,15.4,15.0,10.6,7.7,4.0,0.2,-0.0,0.0,1.1,7.2,11.2,14.3,24.7,18.7,16.8,13.2,12.1,10.3,11.2,10.1,10.3,9.9,10.1,11.4,13.9,14.4,11.0,7.7,4.9,0.5,-0.0,0.0,2.2,7.5,10.6,15.1,24.7,22.2,16.8,13.8,12.3,11.0,12.4,10.6,10.4,10.1,10.2,11.3,15.1,15.0,10.4,7.6,3.6,0.4,0.0,-0.1,1.1,6.1,9.8,12.4,12.3,8.8,6.6,6.3,6.0,5.1,6.4,4.2,4.0,3.9,4.3,5.1,5.1,4.5,6.1,5.3,1.7,-0.1,1.6,-0.6,2.4,4.3,6.5,8.0,9.4,5.8,3.0,4.2,4.5,6.6,6.7,6.6,3.8,6.8,7.0,7.1,7.1,7.5,6.2,3.6,1.6,0.4,2.9,0.1,2.6,4.3,8.7,13.3,14.3,13.2,12.1,12.0,11.3,10.0,10.5,10.8,10.5,10.1,10.2,11.8,14.9,16.3,14.7,12.9,10.6,11.1,10.7,10.9,11.4,9.2,12.3,14.8,24.7,22.2,15.0,13.4,11.5,11.2,11.1,10.0,9.6,9.6,9.9,10.6,13.7,14.5,12.1,9.8,8.3,4.5,3.0,4.9,7.1,8.7,10.4,13.4,15.0,15.4,15.1,11.4,10.5,11.0,8.7,9.7,8.8,9.8,9.6,9.9,13.2,14.7,13.0,12.5,11.1,11.5,10.7,10.3,10.0,10.7,12.3,15.4,17.2,16.6,14.8,14.1,11.9,10.7,12.2,10.7,10.5,10.8,10.6,11.1,13.6,13.9,12.7,10.1,8.5,5.2,1.3,0.5,2.8,7.5,10.6,14.3,20.8,20.4,15.4,13.2,11.9,10.7,10.4,8.9,9.0,9.5,9.4,10.5,13.2,13.4,10.4,7.6,3.6,0.4,-0.0,0.0,1.1,7.5,11.2,14.5,16.4,15.8,14.5,11.4,11.1,10.4,10.2,11.3,11.0,9.3,9.7,10.9,10.1,10.8,10.7,10.0,8.8,8.1,6.6,3.4,5.2,6.4,8.4,11.0,13.7,14.5,12.7,12.1,11.3,8.3,7.8,6.5,3.8,3.6,3.7,3.7,0.8,4.0,2.8,6.2,4.5,3.8,4.5,4.5,2.5,1.5]
```


# Public API
You can find a freely accessible installment of this software [here](https://epexpredictor.batzill.com/).
Get a glimpse of the current prediction [here](https://epexpredictor.batzill.com/prices).

There are no guarantees given whatsoever - it might work for you or not.
I might stop or block this service at any time. Fair use is expected!

# Home Assistant integration
At some point, I might create a HA addon to run everything locally.
For now, you have to either use my server, or run it yourself.



### Configuration:
```yaml
# Make sure you change the parameters fixedPrice and taxPercent according to your electricity plan
sensor:
  - platform: rest
    resource: "https://epexpredictor.batzill.com/prices?country=DE&fixedPrice=13.15&taxPercent=19"
    method: GET
    unique_id: epex_price_prediction
    name: "EPEX Price Prediction"
    scan_interval: 500
    unit_of_measurement: ct/kWh
    value_template: "{{ value_json.prices[0].total }}"
    json_attributes:
      - prices

  # If you want to evaluate performance in real time, you can add another sensor like this
  # and plot it in the same diagram as the actual prediction sensor

  #- platform: rest
  #  resource: "https://epexpredictor.batzill.com/prices?country=DE&fixedPrice=13.15&taxPercent=19&#evaluation=true"
  #  method: GET
  #  unique_id: epex_price_prediction_evaluation
  #  scan_interval: 3600
  #  name: "EPEX Price Prediction Evaluation"
  #  unit_of_measurement: ct/kWh
  #  value_template: "{{ value_json.prices[0].total }}"
  #  json_attributes:
  #    - prices
```

### Display, e.g. via Plotly Graph Card:
```yaml
type: custom:plotly-graph
time_offset: 26h
layout:
  yaxis9:
    fixedrange: true
    visible: false
    minallowed: 0
    maxallowed: 1
entities:
  - entity: sensor.epex_price_prediction
    name: EPEX Price History
    unit_of_measurement: ct/kWh
    texttemplate: "%{y:.0f}"
    mode: lines+text
    textposition: top right
  - entity: sensor.epex_price_prediction
    attribute: dummy
    name: EPEX Price Prediction
    unit_of_measurement: ct/kWh
    texttemplate: "%{y:.0f}"
    mode: lines+text
    textposition: top right
    filters:
      - fn: |-
          ({meta}) => ({
              xs: meta.prices.map(p => new Date(p.startsAt)),
              ys: meta.prices.map(p => p.total)
          })
  - entity: ""
    name: Now
    yaxis: y9
    showlegend: false
    line:
      width: 1
      dash: dot
      color: orange
    x: $ex [Date.now(), Date.now()]
    "y":
      - 0
      - 1
hours_to_show: 30
refresh_interval: 10
```
