# Extraction Functions

This section contains extraction functions that extract `unit` from an input expression. The table below contains the supported units for the given extraction functions.

|   Unit  |                            Description                            |
| :-----: | :---------------------------------------------------------------: |
|   YEAR  |                           extracts year                           |
| QUARTER |                          extracts quarter                         |
|  MONTH  |                           extracts month                          |
|   WEEK  |                     extracts week of the year                     |
|   DAY   |                            extracts day                           |
|   DOY   |                    extract the day of the year                    |
|   DOW   | extract ISO day of the week (ranges from 1 (Monday) to 7 (Sunday) |
|   HOUR  |                           extracts hour                           |
|  MINUTE |                          extracts minute                          |
|  SECOND |                          extracts second                          |



#### <mark style="color:purple;">EXTRACT(</mark>  unit FROM \<datetime expr>  <mark style="color:purple;">)</mark>

Returns the value of specified `unit` from the given date-time expression.

```sql
> select extract(MINUTE FROM cast('2022-01-02 12:27:11' as timestamp))
27
```

#### <mark style="color:purple;">DATEPART(</mark>  unit,  \<expr>  <mark style="color:purple;">)</mark>

Returns the specified unit of the given date/timestamp expression.

```sql
> select current_date, datepart('day', current_date) as day_val;
----------------------------------------
| current_date        |day_val         |
----------------------------------------
| 2023-01-06 00:00:00 |6               |
----------------------------------------
```

#### <mark style="color:purple;">WEEK(</mark> \<date\_or\_timestamp\_expr> <mark style="color:purple;">)</mark>

Returns the week number of the year.

```sql
> select week(FROM_UNIXTIME_WITHUNIT(1673263327000, 'milliseconds'))
2
```

#### <mark style="color:purple;">YEAR(</mark> \<date\_or\_timestamp\_expr> <mark style="color:purple;">)</mark>

Returns the year for a specified date/timestamp.

```sql
> select week(FROM_UNIXTIME_WITHUNIT(1673263327000, 'milliseconds'))
2
```

#### <mark style="color:purple;">LAST\_DAY(</mark>  \<date\_or\_timestamp\_expr>  <mark style="color:purple;">)</mark>

Return the last day of the month for a date or timestamp.

```sql
> select last_day(cast('2022-01-11' as date));
2022-01-31
```





