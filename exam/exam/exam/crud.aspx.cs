using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;
using System.Data;
using System.Data.SqlClient;
using System.Configuration;

namespace exam
{
    public partial class crud : System.Web.UI.Page
    {
        string constr = ConfigurationManager.ConnectionStrings["conn"].ConnectionString;
        protected void Page_Load(object sender, EventArgs e)
        {
            showdata();
        }

        private void showdata()
        {
            try
            {
                SqlConnection cn = new SqlConnection(constr);
                cn.Open();
                SqlCommand cmd = new SqlCommand("select * from demo", cn);
                SqlDataAdapter da = new SqlDataAdapter(cmd);
                DataTable dt = new DataTable();
                da.Fill(dt);
                if (dt.Rows.Count > 0)
                {
                    GridView1.DataSource = dt;
                    GridView1.DataBind();
                }
                else
                {
                    Response.Write("error");
                }
                cn.Close();
            }
            catch (Exception ex)
            {

                throw ex;
            }

        }

        protected void btnins_Click(object sender, EventArgs e)
        {
            insertdata();
        }

        private void insertdata()
        {
            try
            {
                SqlConnection cn = new SqlConnection(constr);
                cn.Open();
                SqlCommand cmd = new SqlCommand("insert into demo(name,address,city)values(@nm,@add,@city)", cn);
                cmd.Parameters.AddWithValue("@nm", txtnm.Text);
                cmd.Parameters.AddWithValue("@add", txtadd.Text);
                cmd.Parameters.AddWithValue("@city", txtct.Text);
                cmd.ExecuteNonQuery();
                cn.Close();
                showdata();
            }
            catch (Exception ex)
            {

                throw ex;
            }

        }

        protected void btnshow_Click(object sender, EventArgs e)
        {
            finddata();
        }

        private void finddata()
        {
            try
            {
                SqlConnection cn = new SqlConnection(constr);
                cn.Open();
                SqlCommand cmd = new SqlCommand("select * from demo where id=@id", cn);
                cmd.Parameters.AddWithValue("@id", txtid.Text);
                SqlDataAdapter da = new SqlDataAdapter(cmd);
                DataTable dt = new DataTable();
                da.Fill(dt);
                if (dt.Rows.Count > 0)
                {
                    txtnm.Text = dt.Rows[0]["name"].ToString();
                    txtadd.Text = dt.Rows[0]["address"].ToString();
                    txtct.Text = dt.Rows[0]["city"].ToString();
                }
                else
                {
                    Response.Write("error");
                }
                cn.Close();
                showdata();
            }
            catch (Exception ex)
            {

                throw ex;
            }
        }

        protected void btnupt_Click(object sender, EventArgs e)
        {
            updatedata();
        }

        private void updatedata()
        {
            try
            {
                SqlConnection cn = new SqlConnection(constr);
                cn.Open();
                SqlCommand cmd = new SqlCommand("update demo set name=@nm,address=@add,city=@ct where id=@id", cn);
                cmd.Parameters.AddWithValue("@nm", txtnm.Text);
                cmd.Parameters.AddWithValue("@add", txtadd.Text);
                cmd.Parameters.AddWithValue("@ct", txtct.Text);
                cmd.Parameters.AddWithValue("@id", txtid.Text);
                cmd.ExecuteNonQuery();
                cn.Close();
                showdata();
            }
            catch (Exception ex)
            {

                throw ex;
            }
        }

        protected void btndlt_Click(object sender, EventArgs e)
        {
            deletedata();
        }

        private void deletedata()
        {
            try
            {
                SqlConnection cn = new SqlConnection(constr);
                cn.Open();
                SqlCommand cmd=new SqlCommand("delete from demo where id=@id",cn);
                cmd.Parameters.AddWithValue("@id",txtid.Text);
                cmd.ExecuteNonQuery();
                cn.Close();
                showdata();
            }
            catch (Exception ex)
            {

                throw ex;
            }
        }
    }
}