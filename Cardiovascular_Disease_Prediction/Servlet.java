/* First Servlet */
import javax.servlet.*;
import
javax.servlet.http.*;
import java.io.*;
public class FirstServlet extends HttpServlet
{
public void init(ServletConfig conf)
{
}
public void doGet(HttpServletRequestreq,HttpServletResponse
res)throws ServletException,IOException
{
String
name=req.getParameter("name");
String
pwd=req.getParameter("pwd");
res.setContentType("text/html");
PrintWriter pw=res.getWriter();
pw.println("<HTML><HEAD>");
pw.println("<TITLE>MyFirst Servlet
Application</TITLE></HEAD>"); pw.println("<BODY>");
pw.println("Hi.... Servlet World Welcomes U. Ms"+name);
pw.println("</BODY></HTML>");
}
public void destroy()
{
}
}