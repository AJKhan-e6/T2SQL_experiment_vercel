# Addition and Subtraction Functions

The table below contains the supported `unit` for the following functions.

|    Unit   | Description |
| :-------: | :---------: |
|  `second` |   Seconds   |
|  `minute` |   Minutes   |
|   `hour`  |    Hours    |
|   `day`   |     Days    |
|   `week`  |    Weeks    |
|  `month`  |    Months   |
| `quarter` |   Quarters  |
|   `year`  |    Years    |



<mark style="color:purple;">**DATE\_ADD(**</mark>**  unit,  value,  \<date\_expr>  **<mark style="color:purple;">**)**</mark>&#x20;

Returns a date expression after adding an interval `value` of type `unit` to the input date expression.

```sql
> select date_add('year', 5, cast('2000-08-05' as date));
2005-08-05
```

#### <mark style="color:purple;">DATE\_DIFF(</mark> \<date expr1>,  \<date\_expr2>  \[,  \<unit>  ]  <mark style="color:purple;">)</mark> &#x20;

Returns the number of units between two date expressions.

`unit` - _optional_ - The unit used to calculate the difference. Default value: day.

```sql
> select date_diff(cast('2006-01-12' as date), cast('2006-01-21' as date));
-9

> select date_diff(cast('2005-10-12' as date), cast('2005-03-12' as date), 'month');
7
```

#### <mark style="color:purple;">TIMESTAMP\_ADD(</mark>  unit,  value,  \<timestamp\_expr>  <mark style="color:purple;">)</mark>&#x20;

Returns a timestamp expression after adding an interval `value` of type `unit` to the input `timestamp expr`.

```sql
> select timestamp_add('hour', 4, cast('2005-10-12 05:10:20' as timestamp));
2005-10-12 09:10:20
```

#### <mark style="color:purple;">TIMESTAMP\_DIFF(</mark>  \<timestamp\_expr1>,  \<timestamp\_expr2>,  unit  <mark style="color:purple;">)</mark> &#x20;

Returns the difference of `timestamp expr1` and `timestamp expr2` with respect to `unit`.

```sql
> select timestamp_diff(cast('2005-10-12 01:25:20' as timestamp), cast('2005-10-12 02:04:13' as timestamp), 'minute');
-38
```



#### <mark style="color:purple;">Date Time Operators</mark>

The following operators can be used to perform mathematical operations on date-time expressions.

#### <mark style="color:purple;">+</mark> - Addition Operator

<pre class="language-sql"><code class="lang-sql">> select current_timestamp + interval 8 minute
2022-09-13 03:13:10

<strong>> select current_date + interval 8 day
</strong>2022-09-21

> select cast('2022-01-02' as date) + interval 24 hour + interval 11 year
</code></pre>

#### <mark style="color:purple;">-</mark> - Subtraction Operator

```sql
> select current_timestamp - interval 2 hour
2022-09-13 03:13:10

> select current_date - interval 8 month
2022-09-21
```

#### Combined Operators

```sql
> select (cast('2022-01-02' as date)  + interval 10 hour) - interval 11 year
2011-01-02
```



**Limits -**

* Week and quarter are not supported in interval (addition/subtraction )

