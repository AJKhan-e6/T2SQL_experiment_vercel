---
description: This document contains the string functions supported by e6data.
---

# String Functions

* [|| or CONCAT- Concatenate](string-functions.md#or-or-concatenation-operator)
* [LIKE](string-functions.md#like-like-operator)
* [CHARACTER\_LENGTH](string-functions.md#character\_length-less-than-expr-greater-than)
  * [CHAR\_LENGTH](string-functions.md#char\_length-less-than-expr-greater-than) _- alias_
  * [LEN ](string-functions.md#len-less-than-expr-greater-than)_- alias_
* [REPLACE](string-functions.md#replace-less-than-expr-greater-than-less-than-search-greater-than-less-than-replace-greater-than)
* [TRIM](string-functions.md#trim-less-than-expr-greater-than)
* [LTRIM](string-functions.md#ltrim-less-than-expr-greater-than)
* [RTRIM](string-functions.md#rtrim-less-than-expr-greater-than)
* [LOWER](string-functions.md#lower-less-than-expr-greater-than)
* [UPPER](string-functions.md#upper-less-than-expr-greater-than)
* [SUBSTRING](string-functions.md#substring-less-than-expr-greater-than-start-length)
  * [SUBSTR](string-functions.md#substr-less-than-expr-greater-than-start-length) _- alias_
* [INITCAP](string-functions.md#initcap-less-than-expr-greater-than)
* [CHARINDEX](string-functions.md#charindex-less-than-expr-greater-than-string-startindex)

#### <mark style="color:purple;">**| |**</mark>** - Concatenation Operator**

returns the final string expression by concating two expressions

```sql
> select 'hello' || 'world';
'helloworld'

> select concat('hello ', 'world')
'hello world'
```

#### <mark style="color:purple;">LIKE</mark> - Like Operator

The `like` operator is used to match a specified pattern in a string expression. The pattern contains regular characters like %,  \_

```sql
expression like pattern 
```

```sql
select colA from table where colA like 'P_%'
```

#### <mark style="color:purple;">CHARACTER\_LENGTH(</mark>  \<expr>  <mark style="color:purple;">)</mark> &#x20;

Returns the length of a given string expression.

```sql
> select character_length('e6data');
6
```

#### <mark style="color:purple;">CHAR\_LENGTH(</mark>  \<expr>  <mark style="color:purple;">)</mark>

This is an alias of the [character\_length](string-functions.md#character\_length-less-than-expr-greater-than) function.

#### <mark style="color:purple;">LEN(</mark>  \<expr>  <mark style="color:purple;">)</mark>

This is an alias of the [character\_length](string-functions.md#character\_length-less-than-expr-greater-than) function.

#### <mark style="color:purple;">REPLACE(</mark> \<expr>,  \<search>  \[,  \<replace>  ]   <mark style="color:purple;">)</mark> &#x20;

`expr` - string expression to be searched/modified.\
`search` - string expression to be searched for in the input expression.\
`replace` - _optional_ string expression to be replaced with the search expression. Default is an empty string.

Replaces all instances of `search` with `replace` in string expression

```sql
> select replace('hard worker', 'hard', 'smart');
'smart worker'

> select replace('hard worker', 'hard');
'worker'
```

#### <mark style="color:purple;">TRIM(</mark>  \<expr>  <mark style="color:purple;">)</mark>

Trims/removes leading and trailing whitespace in the input string expression.

```sql
> select trim('   Fastest database engine  ');
'Fastest database engine'
```

#### <mark style="color:purple;">LTRIM(</mark>  \<expr>  <mark style="color:purple;">)</mark>

Removes leading whitespace in the input string expression.

```sql
> select ltrim('  Fastest database engine--');
'Fastest database engine--'
```

#### <mark style="color:purple;">RTRIM(</mark>  \<expr>  <mark style="color:purple;">)</mark>

Removes trailing whitespace in the input string expression.

```sql
> select rtrim('Fastest database engine  ');
'Fastest database engine'
```

#### <mark style="color:purple;">LOWER(</mark>  \<expr>  <mark style="color:purple;">)</mark>

Returns the input string expression converted to lowercase characters.

```sql
> select lower('FASTEST DataBAse ENgine- 2022');
'fastest database engine- 2022'
```

#### <mark style="color:purple;">UPPER(</mark>  \<expr>  <mark style="color:purple;">)</mark> &#x20;

Returns the input string expression converted to uppercase characters.

```sql
> select upper('Fastest database engine- 2022');
'FASTEST DATABASE ENGINE- 2022'
```

#### <mark style="color:purple;">SUBSTRING(</mark>  \<expr>,  start,  length  <mark style="color:purple;">)</mark>&#x20;

`start` - numeric expression representing a starting position in the string.\
`length` - numeric expression representing the length of the substring.

Returns a substring from an input `string` expression of the given `length` beginning from the `start` index.

_Note: Indexing starts from 1._

```sql
> select substring('Fastest database engine', 9, 8)
'database'
```

#### <mark style="color:purple;">SUBSTR(</mark> \<expr>,   start,  length  <mark style="color:purple;">)</mark>&#x20;

This is an alias of the [substring](string-functions.md#substring-less-than-expr-greater-than-start-length) function

#### <mark style="color:purple;">INITCAP(</mark>  \<expr>  <mark style="color:purple;">)</mark>

Returns string expression with the first letter of each word converted to uppercase.

<pre class="language-sql"><code class="lang-sql">> select initcap('hello world');
<strong>'Hello World'
</strong></code></pre>

#### <mark style="color:purple;">CHARINDEX(</mark>  \<expr>  , string  \[,  startindex]  <mark style="color:purple;">)</mark>

Returns the starting position of the input expression within `string`. Default value of `startindex` is 1.

```sql
> select charindex('a', 'abc')
1

> select charindex('a', 'abc', 2)
0
```

#### <mark style="color:purple;">RIGHT (</mark> \<string><mark style="color:purple;">,</mark> \<offset> <mark style="color:purple;">)</mark> <a href="#right-string-function" id="right-string-function"></a>

Returns the rightmost substring of its input. Offset index starts from 1.&#x20;

Supported datatypes are string, integer, decimal, date, timestamp

```sql
> SELECT right('abc', 2);
a
```

#### <mark style="color:purple;">LEFT (</mark> \<string><mark style="color:purple;">,</mark> \<offset> <mark style="color:purple;">)</mark> <a href="#right-string-function" id="right-string-function"></a>

Returns the leftmost substring of its input. Offset index starts from 1

Supported datatypes are string, integer, decimal, date, timestamp

```sql
> SELECT left('abc', 2);
c
```

#### <mark style="color:purple;">LOCATE (</mark> \<substring> <mark style="color:purple;">,</mark> \<string> _<mark style="color:purple;">\[,</mark> \<start\_position> <mark style="color:purple;">]</mark>_ <mark style="color:purple;"></mark><mark style="color:purple;">)</mark> <a href="#locate-string-function" id="locate-string-function"></a>

Returns the position of the first occurrence of a substring in a string

```sql
> SELECT LOCATE('6', 'e6data');
2
```

```sql
> SELECT LOCATE('6', 'e6data', 3);
0
```

#### <mark style="color:purple;">CONTAINS\_SUBSTR(</mark> \<expr> , \<search\_value\_literal> <mark style="color:purple;">)</mark>

Returns TRUE if the value exists, otherwise returns FALSE.

Supported datatype are, string, integer, decimal, date, timestamp

<pre class="language-sql"><code class="lang-sql"><strong>> SELECT CONTAINS_SUBSTR('abcdef', 'ef');
</strong>TRUE
</code></pre>

**Limitations**

* Complex types are not supported.
* Unicode characters are not supported.

#### <mark style="color:purple;">INSTR(</mark> \<expr>, subvalue _\[, position, occurrence]_ <mark style="color:purple;">)</mark>

Returns the lowest position of subvalue in value.

```sql
> SELECT INSTR('helloooooooo','oo', 1, 3)
9
```

**Limitations**

* Complex types are not supported.

#### <mark style="color:purple;">SOUNDEX(</mark> \<expr> <mark style="color:purple;">)</mark>

Returns a string that contains a phonetic representation of the input string/integer.

```sql
> select soundex('winter')
w536
```
