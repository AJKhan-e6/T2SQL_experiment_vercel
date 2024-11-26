# Regular Expression Functions

#### <mark style="color:purple;">REGEXP\_CONTAINS(</mark> \<expr><mark style="color:purple;">,</mark> \<regexp><mark style="color:purple;">)</mark>

Returns TRUE if the value is a partial match for the regular expression.

Supported data types, string and integer

```sql
> SELECT REGEXP_CONTAINS('abc', '[a-z]{3}');
TRUE
```

#### <mark style="color:purple;">REGEXP\_REPLACE(</mark> \<expr><mark style="color:purple;">,</mark> \<regexp> _<mark style="color:purple;">,</mark> \<replacement>_ <mark style="color:purple;">)</mark>

Returns a STRING where all substrings of value that match regular expression regexp are replaced with replacement. _replacement is optional, by default it is empty._

Supported data types, string and integer

```sql
> SELECT REGEXP_REPLACE('Customers - (NY)','\(|\)','');
Customers-NY
```

#### <mark style="color:purple;">REGEXP\_EXTRACT(</mark> \<expr>,  \<regex pattern>  <mark style="color:purple;">)</mark> &#x20;

Returns the first string from the string `<expr>` that matches `<regex pattern>`.

Supported data types, string and integer

```sql
> SELECT regexp_extract('1a 2b 14m', '\d+')
 1
```

#### <mark style="color:purple;">REGEXP\_EXTRACT\_ALL(</mark> \<expr>,  \<regex pattern>  <mark style="color:purple;">)</mark> &#x20;

Returns an array of all substrings that match the `regexp`. Returns an empty array if there is no match.

Supported data types, string and integer

```sql
> select REGEXP_EXTRACT_ALL('a1_a2a3_a4A5a6', 'a[1-9]');
[a1, a2, a3, a4, a6]
```

#### <mark style="color:purple;">REGEXP\_COUNT (</mark> \<expr>,  \<regex pattern>  <mark style="color:purple;">)</mark> &#x20;

Returns the number of times that a `regexp` pattern occurs in a string.

Supported data types, string and integer

```sql
> select regexp_count('It was the best of times, it was the worst of times', '\bwas\b') as "result";
2 
```



**Note -**&#x20;

Regular expression functions use the [Java pattern](http://docs.oracle.com/javase/8/docs/api/java/util/regex/Pattern.html)

Right now, we have tested on character class and pre-defined character class



**Limitations -**&#x20;

* Complex types such array, struct or map are not supported for querying with regular expressions
