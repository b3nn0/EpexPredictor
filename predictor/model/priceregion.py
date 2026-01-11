from enum import Enum

import holidays


class PriceRegionName(str, Enum):
    """
    Only used for FastAPI so it only offers the pre-defined names
    """
    DE = "DE"
    AT = "AT"
    BE = "BE"
    NL = "NL"

    def to_region(self):
        return PriceRegion[self]


class PriceRegion(Enum):
    country_code: str
    timezone: str
    bidding_zone: str
    latitudes: list[float]
    longitudes: list[float]
    holidays: list[holidays.HolidayBase] # one entry for each regional holiday set, e.g. one for BW, one for BY, ...
    

    def __init__ (self, country_code, timezone, bidding_zone, latitudes, longitudes):
        self.country_code = country_code
        self.bidding_zone = bidding_zone
        self.timezone = timezone
        self.latitudes = latitudes
        self.longitudes = longitudes

        self.holidays = []
        country_holidays = holidays.country_holidays(self.country_code)
        if country_holidays.subdivisions is None or len(country_holidays.subdivisions) == 0:
            self.holidays.append(country_holidays)
        else:
            for subdiv in country_holidays.subdivisions:
                self.holidays.append(holidays.country_holidays(country=self.country_code, subdiv=subdiv))


    DE = (
        "DE",
        "Europe/Berlin",
        "DE-LU",
        [48.4, 49.7, 51.3, 52.8, 53.8, 54.1],
        [9.3, 11.3, 8.6, 12.0, 8.1, 11.6],
    )
    AT = (
        "AT",
        "Europe/Berlin",
        "AT",
        # 24 locations: 2-3 per state covering Vienna, Lower/Upper Austria, Styria, Tyrol, Carinthia, Salzburg, Vorarlberg, Burgenland
        # Vienna (2), Lower Austria (3), Upper Austria (3), Styria (3), Tyrol (3), Carinthia (2), Salzburg (3), Vorarlberg (2), Burgenland (3)
        [48.21, 48.27, 48.31, 48.09, 47.68, 48.31, 48.24, 48.03, 47.07, 47.27, 47.56, 47.26, 47.07, 46.62, 47.80, 47.81, 47.48, 47.30, 47.08, 46.77, 47.62, 47.08, 47.85, 47.31],
        [16.37, 16.17, 15.63, 16.25, 15.44, 14.29, 14.51, 13.93, 15.44, 15.04, 14.29, 13.09, 11.40, 14.31, 13.04, 12.88, 13.38, 10.90, 12.68, 13.37, 14.66, 9.67, 16.53, 16.38],
    )
    BE = (
        "BE",
        "Europe/Berlin",
        "BE",
        # 15 locations: 12 land-based + 3 offshore North Sea wind farm locations
        # Flanders (4): Antwerp, Ghent, Bruges, Leuven
        # Wallonia (4): Liège, Charleroi, Namur, Mons
        # Brussels + extras (4): Brussels, Mechelen, Hasselt, Arlon
        # Offshore North Sea (3): Belgian offshore wind zone
        [51.22, 51.05, 51.21, 50.88, 50.63, 50.41, 50.47, 50.45, 50.85, 51.03, 50.93, 49.68, 51.6, 51.4, 51.7],
        [4.40, 3.72, 3.22, 4.70, 5.57, 4.44, 4.87, 3.95, 4.35, 4.48, 5.33, 5.82, 2.8, 3.0, 2.5],
    )
    NL = (
        "NL",
        "Europe/Amsterdam",
        "NL",
        # 22 locations: 18 land-based + 4 offshore North Sea wind farm locations
        # North: Groningen, Friesland, Drenthe | West: North/South Holland, Utrecht | Central: Gelderland, Flevoland, Overijssel
        # South: North Brabant, Limburg, Zeeland
        # Offshore North Sea (4): Borssele, Hollandse Kust, Northern offshore zones
        [53.22, 53.20, 52.99, 52.37, 52.16, 52.09, 51.99, 52.51, 52.52, 52.42, 51.69, 51.44, 51.56, 51.81, 51.99, 51.44, 51.35, 50.85, 52.5, 52.8, 53.0, 53.3],
        [6.57, 5.79, 6.56, 4.90, 4.50, 5.12, 5.89, 6.08, 5.47, 4.62, 4.78, 5.47, 5.09, 5.84, 4.14, 3.61, 6.17, 5.69, 3.5, 4.2, 4.5, 4.8],
    )


