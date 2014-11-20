/**
 * @file   UtilitiesLib/Error.h
 * @date   Nov 20, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef ERROR_H_
#define ERROR_H_

#include <iostream>

/**
 * @brief Exit with fatal error message
 */
inline void fatalError(std::string const& msg)
{
	std::cout << "FATAL ERROR: " << msg << std::endl;
	exit(1);
}

#endif /* ERROR_H_ */
