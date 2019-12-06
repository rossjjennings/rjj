from dateutil import tz

def human_format(delta):
    """
    Format a datetime.timedelta object in a human-readable way,
    giving a reasonably precise value in days, hours, minutes, and seconds.
    """
    descr = ""
    if delta.days == 1:
        descr += "1 day"
    elif delta.days > 0:
        descr += "{} days".format(days)
    elif delta.days == -2:
        descr += "-1 day"
    elif delta.days < -1:
        descr += "{} days".format(days+1)
    
    seconds = (86399 - delta.seconds) if delta.days < 0 else delta.seconds
    microseconds = (1000000 - delta.microseconds) if delta.days < 0 else delta.microseconds
    hours = seconds//3600
    minutes = (seconds//60) % 60
    seconds = seconds % 60
    if seconds > 0:
        seconds += microseconds/1e6
    
    def add_segment(descr, val, unit_sg, unit_pl):
        if descr != "":
            descr += ", "
        elif delta.days < 0 and val > 0:
            descr += "-"
        
        if val == 1:
            descr += "1 " + unit_sg
        elif val > 0:
            descr += ("{:g} " + unit_pl).format(val)
        
        return descr
    
    descr = add_segment(descr, hours, "hour", "hours")
    descr = add_segment(descr, minutes, "minute", "minutes")
    descr = add_segment(descr, seconds, "second", "seconds")
    if descr == "":
        if microseconds > 1000:
            descr = add_segment(descr, microseconds/1e3, "millisecond", "milliseconds")
        else:
            descr = add_segment(descr, microseconds/1e3, "microsecond", "microseconds")
    
    return descr

def localize(time, zone=''):
    """
    Produce a localized datetime from an astropy Time object.
    Supports Olson database strings such as 'US/Eastern', 'Europe/Lisbon',
    etc. via dateutil.tz. If no time zone is specified, local time is returned.
    """
    dt = time.datetime
    dt = dt.replace(tzinfo=tz.gettz('UTC'))
    return dt.astimezone(tz.gettz(zone))

def localtime(time, zone=''):
    """
    Produce a local time string from an astropy Time object.
    Supports Olson database strings such as 'US/Eastern', 'Europe/Lisbon',
    etc. via dateutil.tz. If no time zone is specified, local time is returned.
    """
    fmt = '%Y-%m-%d %H:%M:%S %Z%z'
    return localize(time, zone).strftime(fmt)
