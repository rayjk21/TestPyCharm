
import pandas as pa
import datedelta
from datetime import timedelta, datetime


def daterange(from_date=None, to_date=None, days=1, weeks=None, months=None, quarters=None, years=None):
#    from_date = '2018-04-05'

    to_date = to_date or datetime.now()
    from_date = from_date or datetime.now()
    if (type(from_date) == str):
        from_date = pa.Timestamp(from_date)

    if years:
        delta = datedelta.datedelta(years=years)
    elif quarters:
        delta = datedelta.datedelta(months=quarters * 3)
    elif months:
        delta = datedelta.datedelta(months=months)
    elif weeks:
        delta = timedelta(weeks=weeks)
    else:
        delta = timedelta(days=days)

    while from_date <= to_date:
        yield from_date
        from_date = from_date + delta
    return

