# review C++ Constructs

This contains a list of reference as well as brief annotations regarding the syntax and semantics of C++ constructs. This is intended to serve as a reviewer.

- pre-processing directives 
    - ifndef/define/else/enfif
    ```
    #ifndef HEADERFILE_H
    /* code */
    #else
    /* code to include if the token is defined */
    #endif
    ```
    or 
    ```
    #ifndef _INCL_GUARD
    #define _INCL_GUARD
    #endif
    ```
    - pragma once

    - helpful documents:
        - https://en.wikipedia.org/wiki/Include_guard
        - https://en.wikipedia.org/wiki/Translation_unit_(programming)
        - https://docs.microsoft.com/en-us/cpp/preprocessor/preprocessor-directives?view=msvc-160
        - https://docs.microsoft.com/en-us/cpp/preprocessor/grammar-summary-c-cpp?view=msvc-160
        - https://stackoverflow.com/questions/1653958/why-are-ifndef-and-define-used-in-c-header-files

- namespaces
    - aliases - You can use an alias declaration to declare a name to use as a synonym for a previously declared type. (This mechanism is also referred to informally as a type alias). You can also use this mechanism to create an alias template, which can be particularly useful for custom allocators.
    ```
    syntax:
    using <identifier> = <type>;
    typedef <type> <identifier>;
    ```
    - note:
        - typedef's are useful for providing clarity with the purpose of your code, not only that it improves the readability of your code but it also allows your code to be easily modified. By the same token, however, it abstracts the underlying types for your variables. It is ideal to use typedefs within a given scope. 

    - helpful documentation:
        - https://docs.microsoft.com/en-us/cpp/cpp/aliases-and-typedefs-cpp?view=msvc-160
        - https://www.cprogramming.com/tutorial/typedef.html#:~:text=The%20typedef%20keyword%20allows%20the,data%20types%20that%20you%20use.

- variables    
    - scopes - delimits the accessibility of your variables
    - local (scoped) static variables -  are variables that live for the entirety of program execution; they
    preserve their value. 
    - helpful documentation:
        - http://www.cplusplus.com/doc/tutorial/variables/

- libraries
    - local
        - dynamic 
        - static
    - exporting libraries 
    - helpful documentation(s):
        - https://docs.microsoft.com/en-us/cpp/build/walkthrough-creating-and-using-a-static-library-cpp?view=msvc-160

- pointers and referneces 
    - initialize `<type>*<identifier> = &<var>`
    - access `*<identifier>`
        - also called as the dereferencing operator 
    - reference `<type>& <identifier>=<referenced var>`
        - you need to declare them upon initialization
    - helpful documentation(s):
        - http://www.cplusplus.com/doc/tutorial/pointers/
        - https://en.cppreference.com/book/pointers
        - https://en.cppreference.com/w/cpp/language/reference
        - pointers https://youtu.be/DTxHyVn0ODg
        - references https://youtu.be/IzoFn3dfsPA

- memory management
    - new
    - delete 
    - helpful documentation(s):

- functions 
    - type of function 
    - passing a function 
    - overriding functions
    - lambda functions 
    - polymorphism 
    - helpful documentation(s):
        - http://www.cplusplus.com/doc/tutorial/functions2/

- structs:: can also have methods and are inherited publicly by default
    
   [When to use Structs over Classes?](https://en.cppreference.com/book/intro/classes)
    - The only technical different between the two is the Structs are public by default while
    classes are private, the reason one should use struct over classes is when grouping values 
    pertaining a set of heterogeneous variables wherein functionalities are only concerned with 
    those values. It should also be noted that some compilers throws a warning for inheriting structs
    and adding more complexity therein. 

    - accessing through pointers
    - constructors and destructors 
    - helpful documentation(s):
        - http://www.cplusplus.com/doc/tutorial/pointers/
        - http://www.cplusplus.com/doc/tutorial/other_data_types/
        - http://www.cplusplus.com/doc/tutorial/structures/

- classes:: inherited privately by default 
    - wrapper class 
    - interface class 
    - inheritance 
    - polymorphism
    - constructors
    - destructors
    - private 
    - public 
    - virtual 
    - abstract class 
    - helpful documentation(s):
        - https://www.tutorialspoint.com/cplusplus/cpp_interfaces.htm
        - http://www.cplusplus.com/doc/tutorial/classes/

- code documentation

- Template
    - helpful documentation(s):
        - http://www.cplusplus.com/doc/tutorial/functions2/

- error handling
    - helpful documentation(s):

- I/O
    - helpful documentation(s):
