# Conversion Functions

#### Convert string to date or timestamp,

```sql
select date '2022-01-02';

select timestamp '2022-01-02';
```

#### <mark style="color:purple;">TO\_TIMESTAMP(</mark>  \<expr>  <mark style="color:purple;">)</mark> &#x20;

Returns the timestamp by parsing the given `string` expression.

<pre class="language-sql"><code class="lang-sql">> select to_timestamp('1997-09-06 12:29:34');
<strong>1997-01-01 12:29:34
</strong></code></pre>

#### <mark style="color:purple;">FROM\_UNIXTIME\_WITHUNIT(</mark>  \<expr>,  \<unit>  <mark style="color:purple;">)</mark>

Returns the UNIX timestamp value as a timestamp. The unit represents the expression value, whether it is in milliseconds or seconds.

* SECONDS - If this keyword unit is specified, the function considers the expression value in seconds
* MILLISECONDS - If this keyword unit is specified, the function considers the expression value in milliseconds

```sql
> select FROM_UNIXTIME_WITHUNIT(1673263327000, 'milliseconds')
'2023-01-09 11:22:07'

> select FROM_UNIXTIME_WITHUNIT(1674797653, 'seconds')
'2023-01-27 05:34:13'
```

#### <mark style="color:purple;">TO\_UNIX\_TIMESTAMP(</mark>  \<expr>  <mark style="color:purple;">)</mark>

Returns the timestamp in `expr` as a UNIX timestamp. Output results will be in epoch milliseconds

```sql
> select to_unix_timestamp(cast('2000-05-08 09:12:10' as timestamp));
957777130000
```



#### <mark style="color:orange;">PARSE FUNCTIONS</mark>

This section contains parse functions that uses the specifier (format\_string). The table below contains the supported patterns for the parse functions.

<table><thead><tr><th width="161">Specifier</th><th>Description</th></tr></thead><tbody><tr><td>%a or %W</td><td>Abbreviated weekday name (Sun .. Sat) or Weekday name (Sunday .. Saturday)</td></tr><tr><td>%b or %M</td><td>Abbreviated month name (Jan .. Dec) or Month name (January .. December)</td></tr><tr><td>%d or %e</td><td>Day of the month, numeric (01 .. 31) or numeric (1 .. 31), this specifier does not support 0 as a month or day.</td></tr><tr><td>%c or %m</td><td>Month, numeric (1 .. 12), this specifier does not support 0 as a month.</td></tr><tr><td>%f</td><td>Fraction of second (6 digits for printing: 000000 .. 999000; 1 - 9 digits for parsing: 0 .. 999999999), timestamp is truncated to milliseconds</td></tr><tr><td>%H or %k</td><td>Hour (00 .. 23)</td></tr><tr><td>%h or %I</td><td>Hour (01 .. 12)</td></tr><tr><td>%j</td><td>Day of year (001 .. 366)</td></tr><tr><td>%i</td><td>Minutes, numeric (00 .. 59)</td></tr><tr><td>%S or %s</td><td>Seconds (00 .. 59)</td></tr><tr><td>%r</td><td>Time of day, 12-hour (equivalent to %h:%i:%s %p)</td></tr><tr><td>%T</td><td>Time of day, 24-hour (equivalent to %H:%i:%s)</td></tr><tr><td>%v</td><td>Week (01 .. 53), where Monday is the first day of the week and we can use %y or %x for year for using %x use 4 digits for year</td></tr><tr><td>%x</td><td>Year for the week, where Monday is the first day of the week, numeric, four digits; used with %v</td></tr><tr><td>%Y</td><td>Year, numeric, four digits</td></tr><tr><td>%y</td><td>Year, numeric (two digits), when parsing, two-digit year format assumes range 1970 .. 2069, so “70” will result in year 1970 but “69” will produce 2069</td></tr><tr><td>%%</td><td>A literal % character</td></tr></tbody></table>

#### <mark style="color:purple;">PARSE\_DATE (</mark> \<format\_string> , \<date\_string> <mark style="color:purple;">)</mark>

Converts a string representation of a date to a DATE object.

```sql
> SELECT PARSE_DATE('%Y/%m/%d/%H','2022/10/20/05');
2022-10-20
```

#### <mark style="color:purple;">PARSE\_DATETIME (</mark> \<format\_string> , \<datetime\_string> <mark style="color:purple;">)</mark>

Converts a string representation of a date-time to a DATETIME object.

```sql
> SELECT PARSE_DATETIME('%a %M %e %Y %H:%i:%s', 'Thu December 25 2008 20:57:10')
25/12/2008 20:57:10
```

#### <mark style="color:purple;">PARSE\_TIMESTAMP (</mark> \<format\_string> , \<timestamp\_string> _\[, \<time\_zone> ]_ <mark style="color:purple;">)</mark>

Converts a string representation of a timestamp to a TIMESTAMP object.

```sql
> SELECT PARSE_TIMESTAMP('%Y-%m-%d %H:%i:%s','2022-01-02 12:27:11','America/New_York')
02/01/2022 17:27:11
```

