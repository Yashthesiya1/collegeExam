<%@ Page Language="C#" AutoEventWireup="true" CodeBehind="crud.aspx.cs" Inherits="exam.crud" %>

<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
    <title></title>
</head>
<body>
    <form id="form1" runat="server">
        <div>
            <table>
                 <tr>
                    <td>id</td>
                    <td>
                        <asp:TextBox runat="server" ID="txtid" />
                    </td>
                </tr>
                <tr>
                    <td>name</td>
                    <td>
                        <asp:TextBox runat="server" ID="txtnm" />
                    </td>
                </tr>
                <tr>
                    <td>address</td>
                    <td>
                        <asp:TextBox runat="server" ID="txtadd" />
                    </td>
                </tr>
                <tr>
                    <td>city</td>
                    <td>
                        <asp:TextBox runat="server" ID="txtct" />
                    </td>
                </tr>
                <tr>
                    <td>
                        <asp:Button Text="insert" runat="server" ID="btnins" OnClick="btnins_Click" />
                        <asp:Button Text="show" runat="server" ID="btnshow" OnClick="btnshow_Click" />
                        <asp:Button Text="update" runat="server" ID="btnupt" OnClick="btnupt_Click" />
                        <asp:Button Text="delete" runat="server" ID="btndlt" OnClick="btndlt_Click" />
                    </td>
            </table>
        </div>
        <asp:GridView ID="GridView1" runat="server" AutoGenerateColumns="False" DataKeyNames="id" >
            <Columns>
                <asp:BoundField DataField="id" HeaderText="id" InsertVisible="False" ReadOnly="True" SortExpression="id" />
                <asp:BoundField DataField="name" HeaderText="name" SortExpression="name" />
                <asp:BoundField DataField="address" HeaderText="address" SortExpression="address" />
                <asp:BoundField DataField="city" HeaderText="city" SortExpression="city" />
            </Columns>
        </asp:GridView>
        <asp:SqlDataSource ID="SqlDataSource1" runat="server" ConnectionString="<%$ ConnectionStrings:examConnectionString3 %>" SelectCommand="SELECT * FROM [demo]"></asp:SqlDataSource>
    </form>
</body>
</html>
