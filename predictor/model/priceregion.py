from enum import Enum
from zoneinfo import ZoneInfo

import holidays

# Central European Time - used by DE, AT, BE
TZ_CENTRAL_EUROPEAN = "Europe/Berlin"


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

    def get_timezone_info(self):
        return ZoneInfo(self.timezone)


    DE = (
        "DE",
        TZ_CENTRAL_EUROPEAN,
        "DE-LU",
        [48.4, 49.7, 51.3, 52.8, 53.8, 54.1],
        [9.3, 11.3, 8.6, 12.0, 8.1, 11.6],
    )
    AT = (
        "AT",
        TZ_CENTRAL_EUROPEAN,
        "AT",
        [48.36, 48.27, 47.32, 47.00, 47.11],
        [16.31, 13.85, 10.82, 13.54, 15.80],
    )
    BE = (
        "BE",
        TZ_CENTRAL_EUROPEAN,
        "BE",
        [51.27, 50.73, 49.99],
        [3.07, 4.79, 5.38],
    )
    NL = (
        "NL",
        "Europe/Amsterdam",
        "NL",
        [52.69, 52.36, 50.51],
        [6.11, 4.90, 5.41],
    )


