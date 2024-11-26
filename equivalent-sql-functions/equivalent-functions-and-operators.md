# Equivalent Functions  & Operators



{% tabs %}
{% tab title="Athena" %}
| e6data                                                         | Athena                                                             |
| -------------------------------------------------------------- | ------------------------------------------------------------------ |
| current\_date()                                                | current\_date                                                      |
| current\_date(timezone)                                        | current\_date AT TIME ZONE timezone                                |
| current\_timestamp()                                           | current\_timestamp                                                 |
| current\_timestamp(timezone)                                   | current\_timestamp AT TIME ZONE timezone                           |
| last\_day( \<date/datetime expr> )                             | date\_trunc('month', date) + interval '1' month - interval '1' day |
| date\_trunc(unit, \<expr> )                                    | DATE\_TRUNC('unit', timestamp)                                     |
| date\_add(unit, value, \<date expr>)                           | date\_add(unit, value, timestamp)                                  |
| date\_diff( \<date expr1>, \<date expr2>, \[\<unit> optional]) | date\_diff(unit, timestamp1, timestamp2)                           |
| timestamp\_add(unit, value, \<timestamp expr>)                 | date\_add(unit, value, timestamp)                                  |
| timestamp\_diff(\<timestamp expr1>, \<timestamp expr2>, unit)  | date\_diff(unit, timestamp1, timestamp2)                           |
| to\_timestamp( \<expr> )                                       | to\_timestamp(string, format)                                      |
| datetime(\<expr>, \<time zone>)                                | \<exp> AT TIME ZONE \<time zone>                                   |
| from\_unixtime(\<expr>, \[\<unit> optional])                   | from\_unixtime(unixtime)                                           |
| to\_unix\_timestamp( \<expr> )                                 | to\_unixtime(expr)                                                 |
| format\_date( \<expr>, format)                                 | format\_datetime(timestamp, format)                                |
| format\_timestamp(\<expr>, format)                             | format\_datetime(timestamp, format)                                |
| extract( unit FROM \<datetime expr>)                           | extract(field FROM x)                                              |
| current\_timestamp + interval '8' minute                       | timestamp '2012-10-31 01:00' + interval '1' month                  |
| array\_to\_string( \<expr>, delimiter)                         | array\_join(x, delimiter, null\_replacement)                       |
| json\_value( \<json expr>, \<json-path>)                       | json\_extract\_scalar(json, json\_path)                            |
{% endtab %}

{% tab title="BigQuery" %}
| e6data                                                         | BigQuery                                                               |
| -------------------------------------------------------------- | ---------------------------------------------------------------------- |
| approx\_quantiles( \<expr>, number)                            | <p>APPROX_QUANTILES(<br>expression, number)</p>                        |
| current\_date()                                                | current\_date()                                                        |
| current\_date(timezone)                                        | CURRENT\_DATE(\[time\_zone])                                           |
| current\_timestamp()                                           | CURRENT\_TIMESTAMP()                                                   |
| current\_timestamp(timezone)                                   | DATETIME(CURRENT\_TIMESTAMP(), "America/Los\_Angeles")                 |
| last\_day( \<date/datetime expr> )                             | last\_day( \<date/datetime expr> )                                     |
| date\_trunc(unit, \<expr> )                                    | DATE\_TRUNC(date\_expression, unit)                                    |
| date\_add(unit, value, \<date expr>)                           | DATE\_ADD(date\_expression, INTERVAL int64\_expression date\_part)     |
| date\_diff( \<date expr1>, \<date expr2>, \[\<unit> optional]) | DATE\_DIFF(date\_expression\_a, date\_expression\_b, date\_part)       |
| timestamp\_add(unit, value, \<timestamp expr>)                 | DATETIME\_ADD(datetime\_expression, INTERVAL int64\_expression part)   |
| timestamp\_diff(\<timestamp expr1>, \<timestamp expr2>, unit)  | DATETIME\_DIFF(datetime\_expression\_a, datetime\_expression\_b, part) |
| to\_timestamp( \<expr> )                                       | TIMESTAMP(string\_expression\[, time\_zone])                           |
| datetime(\<expr>, \<time zone>)                                | DATETIME(timestamp\_expression \[, time\_zone])                        |
| from\_unixtime(\<expr>, \[\<unit> optional])                   | format\_timestamp("%Y-%m-%d", timestamp\_seconds(time))                |
| to\_unix\_timestamp( \<expr> )                                 | UNIX\_SECONDS(expr)                                                    |
| format\_date( \<expr>, format)                                 | FORMAT\_DATE(format\_string, date\_expr)                               |
| format\_timestamp(\<expr>, format)                             | FORMAT\_DATETIME(format\_string, datetime\_expression)                 |
| extract( unit FROM \<datetime expr>)                           | EXTRACT(part FROM date\_expression)                                    |
| current\_timestamp + interval '8' minute                       | current\_timestamp + interval '8' minute                               |
| array\_to\_string( \<expr>, delimiter)                         | ARRAY\_TO\_STRING(array\_expression, delimiter\[, null\_text])         |
| json\_value( \<json expr>, \<json-path>)                       | JSON\_VALUE(json\_expr\[, json\_path])                                 |
{% endtab %}

{% tab title="Databricks" %}
| e6data                                                         | Databricks                                          |
| -------------------------------------------------------------- | --------------------------------------------------- |
| current\_date()                                                | current\_date()                                     |
| current\_date(timezone)                                        | from\_utc\_timestamp(current\_date(),timezone)      |
| current\_timestamp()                                           | current\_timestamp()                                |
| current\_timestamp(timezone)                                   | from\_utc\_timestamp(current\_timestamp(),timezone) |
| last\_day( \<date/datetime expr> )                             | last\_day(expr)                                     |
| date\_trunc(unit, \<expr> )                                    | date\_trunc(unit, expr)                             |
| date\_add(unit, value, \<date expr>)                           | dateadd(unit, value, expr)                          |
| date\_diff( \<date expr1>, \<date expr2>, \[\<unit> optional]) | datediff(endDate, startDate)                        |
| timestamp\_add(unit, value, \<timestamp expr>)                 | timestampadd(unit, value, expr)                     |
| timestamp\_diff(\<timestamp expr1>, \<timestamp expr2>, unit)  | timestampdiff(unit, start, end)                     |
| to\_timestamp( \<expr> )                                       | to\_timestamp(expr \[, fmt] )                       |
| datetime(\<expr>, \<time zone>)                                | from\_utc\_timestamp(expr, timeZone)                |
| from\_unixtime(\<expr>, \[\<unit> optional])                   | from\_unixtime(unixTime \[, fmt])                   |
| to\_unix\_timestamp( \<expr> )                                 | to\_unix\_timestamp(expr \[, fmt] )                 |
| format\_date( \<expr>, format)                                 | date\_format(expr, fmt)                             |
| format\_timestamp(\<expr>, format)                             | date\_format(expr, fmt)                             |
| extract( unit FROM \<datetime expr>)                           | extract(field FROM source)                          |
| datepart( unit, \<expr> )                                      | date\_part(field, expr)                             |
| current\_timestamp + interval '8' minute                       | caldate + interval '1 second'                       |
| array\_to\_string( \<expr>, delimiter)                         | array\_join(array, delimiter \[, nullReplacement])  |
{% endtab %}

{% tab title="Presto" %}
| e6data                                                         | Presto                                            |
| -------------------------------------------------------------- | ------------------------------------------------- |
| current\_date()                                                | current\_date                                     |
| current\_date(timezone)                                        | current\_date AT TIME ZONE timezone               |
| current\_timestamp()                                           | current\_timestamp                                |
| current\_timestamp(timezone)                                   | current\_timestamp AT TIME ZONE timezone          |
| last\_day( \<date/datetime expr> )                             | last\_day\_of\_month(x)                           |
| date\_trunc(unit, \<expr> )                                    | date\_trunc(unit, x)                              |
| date\_add(unit, value, \<date expr>)                           | date\_add(unit, value, timestamp)                 |
| date\_diff( \<date expr1>, \<date expr2>, \[\<unit> optional]) | date\_diff(unit, timestamp1, timestamp2)          |
| timestamp\_add(unit, value, \<timestamp expr>)                 | date\_add(unit, value, timestamp)                 |
| timestamp\_diff(\<timestamp expr1>, \<timestamp expr2>, unit)  | date\_diff(unit, timestamp1, timestamp2)          |
| to\_timestamp( \<expr> )                                       | to\_timestamp(string, format)                     |
| datetime(\<expr>, \<time zone>)                                | \<exp> AT TIME ZONE \<time zone>                  |
| from\_unixtime(\<expr>, \[\<unit> optional])                   | from\_unixtime(unixtime)                          |
| to\_unix\_timestamp( \<expr> )                                 | to\_unixtime(expr)                                |
| format\_date( \<expr>, format)                                 | format\_datetime(timestamp, format)               |
| format\_timestamp(\<expr>, format)                             | format\_datetime(timestamp, format)               |
| extract( unit FROM \<datetime expr>)                           | extract(field FROM x)                             |
| current\_timestamp + interval '8' minute                       | timestamp '2012-10-31 01:00' + interval '1' month |
| array\_to\_string( \<expr>, delimiter)                         | array\_join(x, delimiter, null\_replacement)      |
| json\_value( \<json expr>, \<json-path>)                       | json\_extract\_scalar(json, json\_path)           |
{% endtab %}

{% tab title="Redshift" %}
| e6data                                                         | Redshift                                                                                                 |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| current\_date()                                                | current\_date                                                                                            |
| current\_date(timezone)                                        | current\_date AT TIME ZONE timezone                                                                      |
| current\_timestamp()                                           | current\_timestamp                                                                                       |
| current\_timestamp(timezone)                                   | current\_timestamp AT TIME ZONE timezone                                                                 |
| last\_day( \<date/datetime expr> )                             | LAST\_DAY ( { date \| timestamp } )                                                                      |
| date\_trunc(unit, \<expr> )                                    | DATE\_TRUNC('unit', timestamp)                                                                           |
| date\_add(unit, value, \<date expr>)                           | DATEADD( datepart, interval, {date\|time\|timetz\|timestamp} )                                           |
| date\_diff( \<date expr1>, \<date expr2>, \[\<unit> optional]) | DATEDIFF ( datepart, {date\|time\|timetz\|timestamp}, {date\|time\|timetz\|timestamp} )                  |
| timestamp\_add(unit, value, \<timestamp expr>)                 | DATEADD( datepart, interval, {date\|time\|timetz\|timestamp} )                                           |
| timestamp\_diff(\<timestamp expr1>, \<timestamp expr2>, unit)  | DATEDIFF ( datepart, {date\|time\|timetz\|timestamp}, {date\|time\|timetz\|timestamp} )                  |
| to\_timestamp( \<expr> )                                       | to\_timestamp (timestamp, format)                                                                        |
| datetime(\<expr>, \<time zone>)                                | \<exp> AT TIME ZONE \<time zone>                                                                         |
| from\_unixtime(\<expr>, \[\<unit> optional])                   | timestamp 'epoch' + CAST(your\_timestamp\_column AS BIGINT)/1000 \* interval '1 second'                  |
| to\_unix\_timestamp( \<expr> )                                 | extract(epoch from CURRENT\_DATE)                                                                        |
| format\_date( \<expr>, format)                                 | TO\_CHAR (timestamp\_expression \| numeric\_expression , 'format')                                       |
| format\_timestamp(\<expr>, format)                             | TO\_CHAR (timestamp\_expression \| numeric\_expression , 'format')                                       |
| extract( unit FROM \<datetime expr>)                           | EXTRACT ( datepart FROM { TIMESTAMP 'literal' \| timestamp \| time \| timetz } )                         |
| datepart( unit, \<expr> )                                      | DATE\_PART(datepart, {date\|timestamp})                                                                  |
| current\_timestamp + interval '8' minute                       | caldate + interval '1 second'                                                                            |
| json\_value( \<json expr>, \<json-path>)                       | json\_extract\_path\_text('json\_string', 'path\_elem' \[,'path\_elem'\[, …] ] \[, null\_if\_invalid ] ) |
{% endtab %}

{% tab title="Snowflake" %}
| e6data                                                         | Snowflake                                                                                      |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| current\_date()                                                | current\_date()                                                                                |
| current\_date(timezone)                                        | convert\_timezone('UTC','timestamp', current\_date())                                          |
| current\_timestamp()                                           | current\_timestamp()                                                                           |
| current\_timestamp(timezone)                                   | convert\_timezone('UTC','timestamp', current\_timestamp())                                     |
| last\_day( \<date/datetime expr> )                             | LAST\_DAY( \<date\_or\_time\_expr> \[ , \<date\_part> ] )                                      |
| date\_trunc(unit, \<expr> )                                    | DATE\_TRUNC( \<date\_or\_time\_part>, \<date\_or\_time\_expr> )                                |
| date\_add(unit, value, \<date expr>)                           | DATEADD( \<date\_or\_time\_part>, \<value>, \<date\_or\_time\_expr> )                          |
| date\_diff( \<date expr1>, \<date expr2>, \[\<unit> optional]) | DATEDIFF( \<date\_or\_time\_part>, \<date\_or\_time\_expr1>, \<date\_or\_time\_expr2> )        |
| timestamp\_add(unit, value, \<timestamp expr>)                 | TIMESTAMPADD( \<date\_or\_time\_part> , \<time\_value> , \<date\_or\_time\_expr> )             |
| timestamp\_diff(\<timestamp expr1>, \<timestamp expr2>, unit)  | TIMESTAMPDIFF( \<date\_or\_time\_part> , \<date\_or\_time\_expr1> , \<date\_or\_time\_expr2> ) |
| to\_timestamp( \<expr> )                                       | to\_timestamp( \<expr> )                                                                       |
| datetime(\<expr>, \<time zone>)                                | CONVERT\_TIMEZONE( \<target\_tz> , \<source\_timestamp> )                                      |
| from\_unixtime(\<expr>, \[\<unit> optional])                   | TO\_TIMESTAMP( epoch\_sec )                                                                    |
| to\_unix\_timestamp( \<expr> )                                 | date\_part('epoch\_second',current\_timestamp())                                               |
| format\_date( \<expr>, format)                                 | TO\_VARCHAR( \<date\_or\_time\_expr> \[, '\<format>' ] )                                       |
| format\_timestamp(\<expr>, format)                             | TO\_VARCHAR( \<date\_or\_time\_expr> \[, '\<format>' ] )                                       |
| extract( unit FROM \<datetime expr>)                           | EXTRACT( \<date\_or\_time\_part> FROM \<date\_or\_time\_expr> )                                |
| datepart( unit, \<expr> )                                      | DATE\_PART( \<date\_or\_time\_part> , \<date\_or\_time\_expr> )                                |
| current\_timestamp + interval '8' minute                       | caldate + interval '1 second'                                                                  |
| array\_to\_string( \<expr>, delimiter)                         | ARRAY\_TO\_STRING( \<array> , \<separator\_string> )                                           |
| json\_value( \<json expr>, \<json-path>)                       | JSON\_EXTRACT\_PATH\_TEXT( \<column\_identifier> , '\<path\_name>' )                           |
{% endtab %}
{% endtabs %}

