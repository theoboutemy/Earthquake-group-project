import os
from urllib.parse import urlparse, parse_qs


def getMonths(url):
    """
        Get the start and end months of the given url.

        Args:
            url (String) : Correspond to the url.

         Returns:
            int : This return the year get from the url.
            String : This return the starting month get from the url.
            String : This return the ending month get from the url.

        Raises:
            No raise exception in this function.

        Examples:
            >>> getMonths("url"):
            Result :2023, "01", "06"

        Note:
            The month are stocked as String and not int to be compared after.
    """
    query = parse_qs(urlparse(url).query)
    year = query.get("starttime", [""])[0].split("-")[0]
    month_start = query.get("starttime", [""])[0].split("-")[1]
    month_end = query.get("endtime", [""])[0].split("-")[1]
    return year, month_start, month_end


def writeFile(url, req, namefile, filepath):
    """
        Write the content to a file.

        Args:
            url (String) : Correspond to the url.
            req (String) : Correspond to requests informations.
            namefile (String) : Correspond to the name of the file to create.
            filepath (String) : Correspond to the path to create the file.

        Returns:
            int : This return the status of the function, 0 for success, 1 for failure.

        Raises:
            Exception : if the file cant be created.

        Examples:
            >>> writeFile("url", req, "earthquakes_2023.csv", "ressources/")
            Result : 0

        Note:
            'wb' for the rights of the file.
    """
    try:
        with open(filepath, "wb") as fichier:
            fichier.write(req.content)
        print(f"File '{namefile}' created")
        return 0
    except Exception as e:
        print(f"Error {req.status_code}, cannot create the file for the following url : {url}")
        return 1


def manageFileName(year, month_start, month_end):
    """
        Get the name of the file.

        Args:
            year (int) : Correspond to the year of the file
            month_start (String) : Correspond to the starting month.
            month_end (String) : Correspond to the ending month.

        Returns:
            String : This return the name of the file.

        Raises:
            No raise exception in this function.

        Examples:
            >>> manageFileName(2023, "01" , "12")
            Result : earthquakes_2023.csv

        Note:
            Checking month 6 and 7 to dont make the month 6 two times.
    """
    if month_start == "01" and month_end == "12":
        namefile = f"earthquakes_{year}_P.csv"
    elif month_start == "01" and month_end == "06":
        namefile = f"earthquakes_{year}_P1.csv"
    elif (month_start == "06" or month_start == "07") and month_end == "12":
        namefile = f"earthquakes_{year}_P2.csv"
    elif (month_start == "06" or month_start == "07") and (month_end == "09" or month_end == "08"):
        namefile = f"earthquakes_{year}_P3.csv"
    elif month_start == "09" and month_end == "12":
        namefile = f"earthquakes_{year}_P4.csv"
    else:
        namefile = f"earthquakes_{year}_P.csv"
    return namefile


def createFile(url, req, error_list, folderPath="../ressources/usgs"):
    """
        Create the file with content.

        Args:
            url (String) : Correspond to the url.
            req (String) : Correspond to requests informations.
            error_list (String) : Correspond to the error while generating files.
            folderPath (String) : Correspond to the path of the folder.

        Returns:
            No return value.

        Raises:
            No raise exception in this function.

        Examples:
            >>> createFile("url", req, [], "../ressources/usgs"):
            Result : No return value.

        Note:
            Will create the directory if not already exists.
    """
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    year, month_start, month_end = getMonths(url)
    namefile = manageFileName(year, month_start, month_end)
    filepath = os.path.join(folderPath, namefile)
    if writeFile(url,req, namefile, filepath) == 1:
        error_list.append(namefile)
