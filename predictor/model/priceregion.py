from dataclasses import dataclass
from enum import Enum
from zoneinfo import ZoneInfo

import holidays

# Central European Time - used by DE, AT, BE
TZ_CENTRAL_EUROPEAN = "Europe/Berlin"


PRICE_REGIONS = {}

@dataclass
class PriceRegion:
    country_code: str
    timezone: str
    bidding_zone_energycharts: str | None
    bidding_zone_entsoe: str
    latitudes: list[float]
    longitudes: list[float]

    use_entsoe_load_forecast: bool = True
    use_de_nat_gas_price: bool = True
    holidays: list[holidays.HolidayBase] = None # type:ignore # one entry for each regional holiday set, e.g. one for BW, one for BY, ...
    

    def __post_init__(self):
        self.holidays = []
        country_holidays = holidays.country_holidays(self.country_code)
        if country_holidays.subdivisions is None or len(country_holidays.subdivisions) == 0:
            self.holidays.append(country_holidays)
        else:
            for subdiv in country_holidays.subdivisions:
                self.holidays.append(holidays.country_holidays(country=self.country_code, subdiv=subdiv))

    def get_timezone_info(self):
        return ZoneInfo(self.timezone)


class PriceRegionName(str, Enum):
    """
    Only used for FastAPI so it only offers the pre-defined names
    """
    DE = "DE"
    AT = "AT"
    BE = "BE"
    NL = "NL"
    SE1 = "SE1"
    SE2 = "SE2"
    SE3 = "SE3"
    SE4 = "SE4"
    DK1 = "DK1"
    DK2 = "DK2"

    def to_region(self):
        return PRICE_REGIONS[self]


PRICE_REGIONS[PriceRegionName.DE] = PriceRegion(
    country_code="DE",
    timezone=TZ_CENTRAL_EUROPEAN,
    bidding_zone_energycharts="DE-LU",
    bidding_zone_entsoe="DE_LU",
    latitudes=[48.4, 49.7, 51.3, 52.8, 53.8, 54.1],
    longitudes=[9.3, 11.3, 8.6, 12.0, 8.1, 11.6]
)

PRICE_REGIONS[PriceRegionName.AT] = PriceRegion(
    country_code="AT",
    timezone=TZ_CENTRAL_EUROPEAN,
    bidding_zone_energycharts="AT",
    bidding_zone_entsoe="AT",
    latitudes=[48.36, 48.27, 47.32, 47.00, 47.11],
    longitudes=[16.31, 13.85, 10.82, 13.54, 15.80],
)

PRICE_REGIONS[PriceRegionName.BE] = PriceRegion(
    country_code="BE",
    timezone=TZ_CENTRAL_EUROPEAN,
    bidding_zone_energycharts="BE",
    bidding_zone_entsoe="BE",
    latitudes=[51.27, 50.73, 49.99],
    longitudes=[3.07, 4.79, 5.38],
    use_de_nat_gas_price = False, # doesn't seem to help
)

PRICE_REGIONS[PriceRegionName.NL] = PriceRegion(
    country_code="NL",
    timezone=TZ_CENTRAL_EUROPEAN,
    bidding_zone_energycharts="NL",
    bidding_zone_entsoe="NL",
    latitudes=[52.69, 52.36, 50.51],
    longitudes=[6.11, 4.90, 5.41],
    use_entsoe_load_forecast=False, # Seems to be low quality/reduces performance
    use_de_nat_gas_price = False, # doesn't seem to help
)

PRICE_REGIONS[PriceRegionName.SE1] = PriceRegion(
    country_code="SE",
    timezone=TZ_CENTRAL_EUROPEAN,
    bidding_zone_energycharts=None,
    bidding_zone_entsoe="SE_1",
    latitudes=[65.73, 66.12, 64.98],
    longitudes=[21.50, 22.98, 20.34],
    use_entsoe_load_forecast=False, # not available for SE
    use_de_nat_gas_price = False, # worse performance for SE
)

PRICE_REGIONS[PriceRegionName.SE2] = PriceRegion(
    country_code="SE",
    timezone=TZ_CENTRAL_EUROPEAN,
    bidding_zone_energycharts=None,
    bidding_zone_entsoe="SE_2",
    latitudes=[62.39, 63.01, 61.92],
    longitudes=[17.30, 16.74, 18.14],
    use_entsoe_load_forecast=False, # not available for SE
    use_de_nat_gas_price = False, # worse performance for SE
)

PRICE_REGIONS[PriceRegionName.SE3] = PriceRegion(
    country_code="SE",
    timezone=TZ_CENTRAL_EUROPEAN,
    bidding_zone_energycharts=None,
    bidding_zone_entsoe="SE_3",
    latitudes=[59.34, 60.12, 59.91],
    longitudes=[17.81, 15.78, 16.47],
    use_entsoe_load_forecast=False, # not available for SE
    use_de_nat_gas_price = False, # worse performance for SE
)

PRICE_REGIONS[PriceRegionName.SE4] = PriceRegion(
    country_code="SE",
    timezone=TZ_CENTRAL_EUROPEAN,
    bidding_zone_energycharts="SE4",
    bidding_zone_entsoe="SE_4",
    latitudes=[57.68, 56.24, 55.82],
    longitudes=[12.60, 13.04, 14.10],
    use_entsoe_load_forecast=False, # not available for SE
    use_de_nat_gas_price = False, # worse performance for SE
)

PRICE_REGIONS[PriceRegionName.DK1] = PriceRegion(
    country_code="DK",
    timezone=TZ_CENTRAL_EUROPEAN,
    bidding_zone_energycharts="DK1",
    bidding_zone_entsoe="DK_1",
    latitudes=[57.40, 56.2, 55.38],
    longitudes=[10.24, 8.42, 9.60],
    use_de_nat_gas_price = False, # worse performance for DK
)


PRICE_REGIONS[PriceRegionName.DK2] = PriceRegion(
    country_code="DK",
    timezone=TZ_CENTRAL_EUROPEAN,
    bidding_zone_energycharts="DK2",
    bidding_zone_entsoe="DK_2",
    latitudes=[55.98, 54.91, 55.12],
    longitudes=[12.39, 11.89, 14.73],
    use_de_nat_gas_price = False, # worse performance for DK
)




