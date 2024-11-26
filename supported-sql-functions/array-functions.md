---
description: Array functions supported by e6data
---

# Array Functions

#### <mark style="color:purple;">SUBSCRIPT OPERATOR \[]</mark>

The Subscript operator is used to access the elements of an array. The index starts with 1

<pre class="language-sql"><code class="lang-sql"><strong>> select a[1] from (select array[1,2,3] as a)
</strong>1
</code></pre>

#### <mark style="color:purple;">ELEMENT\_AT (</mark> \<array\_expr>,  index  <mark style="color:purple;">)</mark>

The element\_at function returns element of array at a given index. The index starts with 1

<pre class="language-sql"><code class="lang-sql"><strong>> select element_at(array[1,2,3,4,5],2) 
</strong>2
</code></pre>

#### <mark style="color:purple;">SIZE(</mark> \<array\_expr> <mark style="color:purple;">)</mark>

Returns the size of the input array.

```sql
> select size(array[1,2,3,4,5]) 
5
```

#### <mark style="color:purple;">ARRAY\_TO\_STRING(</mark> \<array\_expr>,  delimiter  <mark style="color:purple;">)</mark>

Concatenates the elements of specified array expression with the provided delimiter.

<pre class="language-sql"><code class="lang-sql"><strong>> select array_to_string(array[1,2,3], '+')
</strong>'1+2+3'
</code></pre>

####

####

#### <mark style="color:purple;">GENERATE\_DATE\_ARRAY(<</mark>start\_date/timestamp>, \<end\_date/timestamp> , INTERVAL \<step\_int> \<date\_part>)

Returns an array of date.&#x20;

Supported date\_part must be either DAY, MONTH, YEAR

The step\_int parameter determines the increment used to generate dates.

<pre class="language-sql"><code class="lang-sql"><strong>> SELECT GENERATE_DATE_ARRAY( '2023-03-04', '2023-04-26' , INTERVAL 1 DAY) AS date_array
</strong>[2023-03-04, 2023-03-05, 2023-03-06]
</code></pre>

#### <mark style="color:purple;">GENERATE\_TIMESTAMP\_ARRAY(<</mark>start\_timestamp>, \<end\_timestamp>, INTERVAL \<step\_int>  \<date\_part>)

Returns an array of timestamp.&#x20;

Supported date\_part must be either DAY, HOUR, MINUTE, SECOND

The step\_int parameter determines the increment used to generate timestamps.

<pre class="language-sql"><code class="lang-sql"><strong>> SELECT GENERATE_TIMESTAMP_ARRAY(to_timestamp('2016-10-05 00:00:00'), to_timestamp('2016-10-05 05:00:00'),
</strong>                                INTERVAL 1 hour) AS timestamp_array
[2016-10-05 00:00:00, 2016-10-05 01:00:00, 2016-10-05 02:00:00, 2016-10-05 03:00:00, 2016-10-05 04:00:00, 2016-10-05 05:00:00]
</code></pre>
